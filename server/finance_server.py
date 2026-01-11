from mcp.server.fastmcp import FastMCP
import yfinance as yf
import json
from datetime import datetime
from groq import Groq
import requests
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup


load_dotenv()
groq_api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq(api_key=groq_api_key)
mcp = FastMCP("finance")


def _fetch_text(url: str) -> str:
    """内部辅助函数：纯粹的爬虫逻辑"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # 1. 发送请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 2. 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. 提取正文 (针对 Yahoo Finance 和一般网站的通用逻辑)
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # 获取所有段落文本
        paragraphs = soup.find_all('p')
        text_content = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 10] # 过滤掉太短的行
        
        full_text = "\n\n".join(text_content)
        
        # 截断过长的文本以节省 Token
        if len(full_text) > 12000:
            full_text = full_text[:12000] + "\n...(content truncated)..."
            
        if not full_text:
            return "Error: Unable to extract text content from this URL."
            
        return f"Source URL: {url}\n\nContent:\n{full_text}"

    except Exception as e:
        return f"Error fetching article: {str(e)}"


@mcp.tool()
def get_stock_data(ticker: str, prepost: bool = False) -> str:
    """
    获取股票或大盘的实时价格和基础信息。
    Args:
        ticker: 股票代码 (例如: 'NVDA', 'AAPL') 或 大盘代码 (例如: '^GSPC' 代表标普500, '^IXIC' 代表纳指)
        prepost: 是否获取盘前价格信息。早报应设为True，晚报时设为False
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 获取最新即时数据 (1天内的历史数据，取最后一行)
        hist = stock.history(period="1d", interval="30min", prepost=True)
        
        if hist.empty:
            return f"Error: No data found for ticker {ticker}"
        
        current_price = hist['Close'].iloc[-1]
        open_price = hist['Open'].iloc[-1]
        change_percent = ((current_price - open_price) / open_price) * 100
        
        # 尝试获取部分基础信息（如果是大盘，info可能较少）
        info = stock.info
        name = info.get('shortName', ticker)
        market_cap = info.get('marketCap', 'N/A')
        
        data = {
            "symbol": ticker,
            "name": name,
            "current_price": round(current_price, 2),
            "daily_change_percent": round(change_percent, 2),
            "market_cap": market_cap,
            "currency": info.get('currency', 'USD'),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return json.dumps(data, ensure_ascii=False)
        
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {str(e)}"


@mcp.tool()
def get_market_news(ticker: str, limit: int = 8) -> str:
    """
    获取指定股票或大盘的最新相关新闻。
    Args:
        ticker: 股票代码 (例如: 'NVDA', 'AAPL') 或 大盘代码 (例如: '^GSPC' 代表标普500, '^IXIC' 代表纳指)
        limit: 返回的新闻数量，默认8条
    """
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        
        if not news_list:
            return f"No news found for {ticker}."
            
        processed_news = []
        for item in news_list[:limit]:
            processed_news.append({
                "title": item.get('title'),
                "publisher": item.get('publisher'),
                "link": item.get('link'),
                "publish_time": datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M')
            })
            
        return json.dumps(processed_news, ensure_ascii=False)
        
    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


@mcp.tool()
def summarize_news(url: str, focus_instruction: str = "General summary") -> str:
    """
    阅读指定 URL 的新闻，并使用 17B 模型根据指示进行总结。
    
    Args:
        url: 新闻链接
        focus_instruction: 给总结模型的具体指令。例如："关注英伟达的GPU销量数据"。如果没有特别要求默认为"General summary"。
    """
    # 1. 获取原文
    raw_text = _fetch_text(url)
    if not raw_text:
        return "Error: Failed to fetch content from URL."

    # 2. 构建 Prompt
    system_prompt = (
        "You are a high-efficiency financial news extractor. Your output acts as a data feed for a senior analyst with limited bandwidth. "
        "You must compress the article content based on the user's specific instruction into the following strict format:\n\n"
        
        "### 1. EXECUTIVE SUMMARY\n"
        "- Provide a dense, high-level overview.\n"
        "- Focus strictly on the user's instruction (e.g., if asked about 'revenue', ignore 'product design').\n\n"
        
        "### 2. HARD DATA (Output 'None' if missing)\n"
        "- List ONLY specific numbers, percentages, currency values, dates, or ticker changes.\n"
        "- Format: `[Metric]: [Value]` (e.g., 'Revenue increase by: $1.4B', 'EPS: $2.12').\n\n"
        
        "### 3. KEY QUOTES (Output 'None' if missing)\n"
        "- Extract 1-2 most critical direct quotes from decision-makers (CEO, CFO, Analysts).\n\n"
        
        "CRITICAL CONSTRAINTS:\n"
        "- Keep the summary between 350 and 450 words.\n"
        "- Do not use fluff or filler words. Be telegraphic.\n"
    )
    
    user_prompt = f"""
    --- ARTICLE CONTENT ---
    {raw_text}
    -----------------------
    
    User INSTRUCTION: {focus_instruction}
    
    Please summarize the article above, strictly following the instruction.
    """

    try:
        # 3. 调用 8B 模型进行总结
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1, # 事实提取需要低温度
        )
        
        summary = chat_completion.choices[0].message.content
        return f"Source: {url}\nFocus: {focus_instruction}\n\nSummary:\n{summary}"

    except Exception as e:
        return f"Error during summarization: {str(e)}"


if __name__ == "__main__":
    mcp.run()