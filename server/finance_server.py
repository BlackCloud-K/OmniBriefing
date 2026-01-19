import concurrent.futures
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import json
from datetime import datetime
from groq import Groq
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import trafilatura
from time import sleep

# 1. 初始化环境
load_dotenv()
# 注意：保持你原有的设置，使用 OPENAI_API_KEY 变量名读取 Groq Key
groq_api_key = os.getenv("OPENAI_API_KEY") 
if not groq_api_key:
    print("Warning: OPENAI_API_KEY (for Groq) not found.")

groq_client = Groq(api_key=groq_api_key)
mcp = FastMCP("finance")

# === 🌟 核心升级: 全局会话状态 (The Session State) ===
# 这就像一个“购物车”，用来暂存 Agent 挑选的数据
SESSION_STATE = {
    "prices": {},       # 存股价: {"NVDA": {...}, "AAPL": {...}}
    "raw_news": [],     # 存原始新闻: [{"id": 0, "title": "...", "url": "...", "ticker": "..."}]
    "summaries": []     # 存总结好的新闻: [{"id": 0, "summary": "..."}]
}

def _reset_session():
    """清空购物车，开始新的一轮分析"""
    SESSION_STATE["prices"] = {}
    SESSION_STATE["raw_news"] = []
    SESSION_STATE["summaries"] = []

# === 2. 爬虫工具 (保留你现有的 Trafilatura 逻辑) ===
def _fetch_text(url: str) -> str:
    """
    使用 trafilatura 库进行本地智能提取。
    """
    try:
        # 1. 下载 (它会自动处理 User-Agent 和简单的反爬重试)
        downloaded = trafilatura.fetch_url(url)
        
        if not downloaded:
            return "Error: Failed to download page."
            
        # 2. 提取 (智能识别正文，忽略侧边栏和广告)
        text = trafilatura.extract(
            downloaded, 
            include_comments=False, 
            include_tables=True,
            no_fallback=True
        )
        
        if not text or len(text) < 200:
            return "Error: Extracted content empty or too short."
            
        return text

    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 🛒 Tool 1: 存股价 (Add Prices to Cart)
# ==========================================
@mcp.tool()
def fetch_and_store_prices(tickers: list[str], prepost: bool = True) -> str:
    """
    Fetch and store stock prices for given ticker symbols.
    
    Args:
        tickers: A list of stock ticker symbols (e.g., ["AAPL", "NVDA", "TSLA"]).
        prepost: Optional boolean (default: False). If True, includes pre-market and post-market data.
                Set to True if you need extended hours trading data.
    """
    _reset_session() # 视为新会话开始，清空旧数据
    
    if not tickers:
        return "No tickers provided."

    # 定义单个抓取逻辑 (复用你之前的逻辑)
    def fetch_single_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            # 策略: 优先取 1天，如果是周末/休市取不到，则回退取 5天
            hist = stock.history(period="1d", interval="1h", prepost=prepost)
            if hist.empty:
                hist = stock.history(period="5d", interval="1h", prepost=prepost)
            
            if hist.empty:
                return {"symbol": ticker, "status": "No Data", "error": "Market Closed/No Data"}
            
            current_price = hist['Close'].iloc[-1]
            
            # 计算涨跌幅
            last_date = hist.index[-1].date()
            day_data = hist[hist.index.date == last_date]
            
            if not day_data.empty:
                open_price = day_data['Open'].iloc[0]
            else:
                open_price = hist['Open'].iloc[-1]
            
            info = {}
            try: info = stock.info
            except: pass

            prev_close = info.get('previousClose')
            base_price = prev_close if prev_close else open_price
            
            change_percent = ((current_price - base_price) / base_price) * 100
            name = info.get('shortName', info.get('longName', ticker))
            
            return {
                "symbol": ticker,
                "name": name,
                "price": round(current_price, 2),
                "change": round(change_percent, 2),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": 'Active',
                "price_history": hist['Close']
            }
        except Exception as e:
            return {"symbol": ticker, "status": "Error", "error": str(e)}

    # 并发执行
    results_summary = []
    max_workers = min(len(tickers), 10)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker, t): t for t in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            ticker = data["symbol"]
            
            # 存入全局 Session
            SESSION_STATE["prices"][ticker] = data
            
            # 生成简报字符串返回给 Client
            if data["status"] == "Active":
                results_summary.append(f"{ticker}: {data['change']}%")
            else:
                results_summary.append(f"{ticker}: {data['status']}")
            
    return f"Prices stored in server. Quick View: {', '.join(results_summary)}"

# ==========================================
# 🛒 Tool 2: 查新闻菜单 (Search & Menu)
# ==========================================
@mcp.tool()
def search_news_options(tickers: list[str], limit: int = 4) -> str:
    """
    Search and retrieve news article options for given stock tickers.
    
    Args:
        tickers: A list of stock ticker symbols (e.g., ["AAPL", "NVDA", "TSLA"]).
                News will be searched for each ticker symbol provided.
        limit: Optional integer (No more than 4). Maximum number of news articles to retrieve per ticker.
               Higher values return more articles but may take longer to process.
    """
    if not tickers:
        return "No tickers provided."
        
    SESSION_STATE["raw_news"] = [] # 清空旧新闻列表
    global_index = 0
    menu_output = []
    
    # 用于去重的集合：跟踪已见过的 URL 和标题
    seen_urls = set()
    seen_titles = set()
    
    # 内部函数：获取单只股票新闻
    def fetch_single_news(ticker):
        try:
            stock = yf.Ticker(ticker)
            news_list = stock.news
            if not news_list: return []
            
            valid_items = []
            safe_limit = min(limit, len(news_list))
            
            for item in news_list[:safe_limit]:
                # 复用你的解析逻辑
                data = item.get('content', item)
                title = data.get('title', 'No Title')
                
                # 提取链接
                link = None
                if 'clickThroughUrl' in data and data['clickThroughUrl']:
                    link = data['clickThroughUrl'].get('url')
                if not link and 'canonicalUrl' in data and data['canonicalUrl']:
                    link = data['canonicalUrl'].get('url')
                if not link:
                    link = data.get('link') or data.get('url')
                    
                if link and title != "No Title":
                    valid_items.append({"ticker": ticker, "title": title, "url": link})
            return valid_items
        except:
            return []

    # 并发抓取新闻元数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tickers), 10)) as executor:
        future_to_ticker = {executor.submit(fetch_single_news, t): t for t in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            items = future.result()
            for item in items:
                # 去重检查：如果 URL 或标题已存在，则跳过
                url = item["url"]
                title = item["title"]
                
                # 标准化 URL 和标题用于比较（去除首尾空格，转为小写）
                url_normalized = url.strip().lower() if url else ""
                title_normalized = title.strip().lower() if title else ""
                
                # 如果 URL 或标题已存在，跳过这条新闻
                if url_normalized in seen_urls or title_normalized in seen_titles:
                    continue
                
                # 添加到已见集合
                if url_normalized:
                    seen_urls.add(url_normalized)
                if title_normalized:
                    seen_titles.add(title_normalized)
                
                # 存入全局列表，分配 ID
                entry = {
                    "id": global_index,
                    "ticker": item["ticker"],
                    "title": item["title"],
                    "url": item["url"]
                }
                SESSION_STATE["raw_news"].append(entry)
                
                # 生成菜单项
                menu_output.append(f"[{global_index}] {item['ticker']} | {item['title']}")
                global_index += 1
    
    if not menu_output:
        return "No news found."
        
    return "Available News Options (Select by ID):\n" + "\n".join(menu_output)

# ==========================================
# 🛒 Tool 3: 选新闻并总结 (Checkout)
# ==========================================
@mcp.tool()
def summarize_selected_indices(indices: list[int], focus_instruction: str = "General summary") -> str:
    """
    Fetch and summarize selected news articles by their indices.
    
    Args:
        indices: A list of integer indices corresponding to news articles from search_news_options.
                For example, [0, 2, 5] will summarize the articles at positions 0, 2, and 5.
                Indices must be valid (within the range of available news articles).
        focus_instruction: Optional string (default: "General summary"). Custom instruction for the AI
                          summarization process.
    """
    selected_items = []
    # 验证 ID
    for idx in indices:
        if 0 <= idx < len(SESSION_STATE["raw_news"]):
            selected_items.append(SESSION_STATE["raw_news"][idx])
            
    if not selected_items:
        return "Invalid indices provided."

    print(f"Summarizing {len(selected_items)} selected articles...")

    # 内部处理函数
    def process_item(item):
        url = item['url']
        ticker = item['ticker']
        
        # 1. 抓取
        raw_text = _fetch_text(url)
        if not raw_text or raw_text.startswith("Error"):
            return {
                "id": item['id'],
                "ticker": ticker,
                "summary": f"Failed to fetch content: {raw_text}"
            }

        # 2. 总结 (使用 Groq 17B)
        system_prompt = (
            "You are a high-efficiency financial news extractor. "
            "Compress the article content into strict format:\n"
            "### 1. EXECUTIVE SUMMARY\n"
            "### 2. HARD DATA (Numbers/Dates)\n"
            "### 3. KEY QUOTES\n"
            "Constraints: Under 400 words. Be telegraphic."
        )
        user_prompt = f"User INSTRUCTION: {focus_instruction}\n\nCONTENT:\n{raw_text[:12000]}"

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.1,
            )
            summary = chat_completion.choices[0].message.content
            return {
                "id": item['id'],
                "ticker": ticker,
                "title": item['title'],
                "summary": summary
            }
        except Exception as e:
            return {"id": item['id'], "ticker": ticker, "summary": f"Error: {str(e)}"}

    # 并发总结
    new_summaries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_item, item) for item in selected_items]
        for future in concurrent.futures.as_completed(futures):
            new_summaries.append(future.result())
            
    # 存入 Session (追加模式)
    SESSION_STATE["summaries"].extend(new_summaries)
    
    # 返回 JSON 给 Agent，方便它决定下一步
    return json.dumps(new_summaries, ensure_ascii=False)

# ==========================================
# 🛒 Tool 4: 删除新闻 (Remove News)
# ==========================================
@mcp.tool()
def remove_news_summaries(indices: list[int]) -> str:
    """
    Remove news summaries by their indices and return remaining indices.
    
    Args:
        indices: A list of integer indices to remove from stored news summaries.
                These indices correspond to the 'id' field in the summaries.
    
    Returns:
        A JSON string containing the list of remaining summary indices after removal.
    """
    if not indices:
        return json.dumps([item['id'] for item in SESSION_STATE["summaries"]], ensure_ascii=False)
    
    # 删除指定 indices 的新闻
    indices_to_remove = set(indices)
    SESSION_STATE["summaries"] = [
        item for item in SESSION_STATE["summaries"] 
        if item['id'] not in indices_to_remove
    ]
    
    # 返回剩余 indices
    remaining_indices = [item['id'] for item in SESSION_STATE["summaries"]]
    return json.dumps(remaining_indices, ensure_ascii=False)

# ==========================================
# 🛒 Tool 5: 导出报告 (Export)
# ==========================================
@mcp.tool()
def export_final_report() -> str:
    """
    Generate a final Markdown-formatted market report.
    
    Args:
        (No parameters)
    """
    sleep(50)
    md = "# Daily Market Pulse\n\n"
    
    # 1. 股价部分
    md += "## Market Data\n"
    for ticker, data in SESSION_STATE["prices"].items():
        if data.get("status") == "Active":
            # 1. 基础信息
            md += f"- **{ticker}**: {data['name']} ${data['price']} ({data['change']}%)\n"
            
            # 2. 清洗并格式化分时数据 (Intraday Trend)
            history = data.get("price_history")
            
            # 检查 history 是否是 Pandas Series (因为有时候可能存成 list)
            if hasattr(history, 'index') and not history.empty:
                # 列表推导式：只取 "时:分" 和 "价格"
                trend_points = [
                    f"{t.strftime('%H:%M')}:${p:.2f}" 
                    for t, p in zip(history.index, history.values)
                ]
                # 用箭头连接，既紧凑又直观
                trend_line = " → ".join(trend_points)
                md += f"  - *Price Trend*: {trend_line}\n"
        else:
            md += f"- **{ticker}**: {data.get('status')}\n"
            
    # 2. 新闻部分
    md += "\n## Key Developments\n"
    if not SESSION_STATE["summaries"]:
        md += "(No news selected)\n"
    
    for item in SESSION_STATE["summaries"]:
        md += f"\n### [{item['ticker']}] {item.get('title', 'News')}\n"
        md += f"{item['summary']}\n"
        md += f"*(Ref ID: {item['id']})*\n"
        
    return md

if __name__ == "__main__":
    mcp.run()