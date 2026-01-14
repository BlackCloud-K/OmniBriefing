import asyncio
from json import load
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from mcp_agent import FastAgent, RequestParams
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests
from groq import Groq
import google.genai as genai

load_dotenv()
client = FastAgent("Finance Client")

@client.agent(
    name = "OmniBriefing - Finance Client",
    instruction= """
        You are a High-Precision Financial Information Curator. Your job is NOT to write articles, but to aggregate, filter, and list raw intelligence.

        ### WORKFLOW PROTOCOL

        **PHASE 1: DATA ACQUISITION (The Net)**
        1. **Time Check**: 
        - Weekdays: Fetch Price + News.
        - Weekends: Fetch News ONLY.
        2. **Target List**:
        - **Indices**: ^GSPC, ^DJI, ^IXIC.
        - **Tech**: [User's Watchlist].
        - **Industrial**: [User's Watchlist] (Filter: Ignore if absolute change < 1.5% AND no major news keywords).
        - **Commodities**: [User's Watchlist] (Gold, Silver, Oil, Copper, Natural Gas, etc.)
        3. **Action**: 
        - **CRITICAL**: When calling tools, use the EXACT tool names: `fetch_and_store_prices`, `search_news_options`, `summarize_selected_indices`, `remove_news_summaries`, `export_final_report`.
        - **Parameter Format**: 
          * `tickers` must be a list of strings: `["^GSPC", "^DJI", "NVDA"]`
          * `limit` must be an integer: `3`
          * `indices` must be a list of integers: `[0, 1, 2]`

        **PHASE 2: URL SELECTION (The Filter)**
        1. Review the news menu from Phase 1.
        2. **Selection Logic**: Select indices for processing ONLY IF:
        - The headline contains high-impact keywords: "Earnings", "Guidance", "Acquisition", "Layoffs", "CEO", "FDA", "Lawsuit", "Breakthrough".
        - OR the stock price moved > 3% (Volatility Check).
        - *Constraint*: Maximum 2 articles per company. If news is generic/repetitive, skip it.
        3. **Batching**: Compile ALL selected indices into a single list.
        4. **Action**: Call `summarize_selected_indices(indices=[...], focus_instruction="...")` ONCE with all selected indices.

        **PHASE 3: QUALITY CONTROL & REFINEMENT (The Critic)**
        *This is an iterative process. Do not blindly accept the first result.*

        1. **Review**: Analyze the summaries received from Phase 2.
        2. **Evaluation Criteria (Is it "OK"?):**
        - **Low Quality**: If the news is spam (e.g., "Law firm class action reminder") or generic. -> **Action**: Use `remove_news_summaries(indices=[...])` to delete these items.
        - **Redundancy**: If multiple items cover the exact same event. -> **Action**: Keep the most detailed one, use `remove_news_summaries(indices=[...])` to delete the rest.
        3. **Deletion Process**:
        - Identify the indices (id field) of news items to remove.
        - **CRITICAL CONSTRAINT**: You MUST keep at least 4 news summaries. Do NOT delete news if it would result in fewer than 5 summaries remaining.
        - Call `remove_news_summaries(indices=[...])` with the list of indices to delete, but only if at least 5 summaries will remain.
        - The tool will return the remaining indices after deletion.
        4. **Loop**: 
        - Repeat review and deletion until you are satisfied that the report explains the major market movements.
        - **Remember**: Always maintain at least 4 news summaries in the final report.

        **PHASE 4: Task Completion & Report Generation**
        *Only proceed here when the content is fully curated and ready.*
        *1. Print `-------` as a separator before calling export_final_report.*
        *2. Call `export_final_report()`.*
        *3. Print `-------` again after receiving the report result.*
        *4. Your task is COMPLETE. End your response immediately after printing the second separator.*
        *Note: You do NOT need to read, process, or summarize the report content. Just call the tool, print the separator, and exit.*

        If "Failed to call any function", please at least retry 1 time.
    """,
    servers = ["finance"],
    # 使用 Groq 模型，格式：groq.model-name
    # 注意：需要确保 OPENAI_API_KEY 环境变量设置为 Groq API key（与 server 端一致）
    model="groq.llama-3.3-70b-versatile",
    #model="groq.meta-llama/llama-4-scout-17b-16e-instruct",
    request_params=RequestParams(
        max_iterations=15,
    )
)


async def analyze_report(md_file_path: str) -> str:
    """
    使用 Gemini 模型直接分析 markdown 报告内容。
    
    Args:
        md_file_path: markdown 文件路径
    
    Returns:
        分析结果字符串
    """
    # 读取 markdown 文件内容
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            report_content = f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"
    
    # 移除第一行的日期行（如果存在）
    if report_content.startswith("Generated on:"):
        lines = report_content.split("\n", 1)
        if len(lines) > 1:
            report_content = lines[1].strip()
    
    # 从环境变量读取 Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        return "错误：未找到 GEMINI_API_KEY 或 GOOGLE_API_KEY 环境变量"
    
    # 创建客户端
    client = genai.Client(api_key=gemini_api_key)
    
    # 选择模型
    model = 'gemini-3-flash-preview'  # 或 'gemini-pro'，根据需要选择
    
    # 构建分析提示
    analysis_prompt = f"""
你是一位拥有 20 年经验的华尔街**首席交易员 (Chief Trader)**。你面对的不是普通散户，而是机构投资者。
你的任务是基于提供的《市场原始数据》和《新闻简报》，撰写一份**深度复盘报告**。

### 核心思维模型：透过现象看本质 (Connect the Dots)

#### 1. 分时微观结构分析 (Price Action is King)
你拥有每只股票的 `Price Trend` (分时走势字符串)。**你必须分析全天的价格路径，而不仅仅是收盘价！**
* **V型反转?** (早盘大跌，午盘拉回) -> 暗示下方有强力支撑，空头陷阱。
* **单边下行?** (开盘即最高，一路阴跌) -> 暗示卖压沉重，多头无力抵抗。
* **尾盘异动?** (全天横盘，最后30分钟突袭) -> 暗示有内幕资金博弈隔夜消息或ETF调仓。
* **写作要求**：在分析个股时，必须明确引用 Trend 中的时间点和价格变化来支撑你的观点。

#### 2. 板块内部分化 (Sector Divergence) **(关键升级)**
* **拒绝“一刀切”**：不要简单地说“科技股跌了”。
* **寻找 Alpha**：如果同板块中（如 Google vs MSFT，或 AMD vs NVDA）出现走势背离，这是最重要的市场信号。
    * *必须分析*：为什么大盘跌它却涨？是基本面独有某种利好？还是避险属性？
    * *对比视角*：在复盘个股时，尝试引入同行业龙头的对比视角。

#### 3. 深度归因与逻辑推演
* **寻找背离 (Price vs. News)**：如果有股票在 `Key Developments` 里全是利空，但 `Price` 却涨了 -> **这意味着利空出尽 (Priced-in)**。你必须指出这种异常。
* **语言精确度 (Nuance)**：由于你缺乏成交量 (Volume) 和 挂单 (Level 2) 数据，**严禁使用绝对化词汇**（如“机构正在抛售”）。
    * *请使用*：“价格形态**暗示**机构减持”、“缺乏买盘承接”、“可能是获利了结”。

#### 4. 信息不足时的处理 (Info Gaps)
* 如果某只股票涨跌幅巨大 (例如 >3%)，但提供的 `Key Developments` 里完全没有关于它的新闻：
    * **严禁瞎编原因！**
    * **必须输出**：
        1. 标注 **【待确认】**。
        2. 明确写出：“当前简报缺乏直接解释。”
        3. **提供搜索建议**：给出 2-3 个具体的搜索关键词 (如 "NG=F inventory data delay", "Company X M&A rumors")，方便人类交易员进行后续排查。

---

### 输出格式 (Markdown)

#### 1. 交易员视角 (Trader's View)
* **今日盘面特征和开篇总结**: (用1-3句话总结)
* **分时异动**: 分析各个领域（科技、制造、大宗商品黄金等）4-5只有代表性的股票价格
    * [股票代码+股票名称]: 结合Trend数据给出分析（重点关注开盘、午盘转折、收盘）

#### 2. 深度个股复盘 (Deep Dive)
* **[代码+名称] (涨跌幅)**: 
    * *基本面*: (结合新闻，**若有同业对比请在此处强调**)
    * *技术面*: (结合 Trend 数据，分析买卖力道)
    * *机构意图*: (使用概率性语言推测：洗盘、出货、避险或建仓)

#### 3. 异常与待确认 (Watchlist & Gaps)
* 列出那些波动剧烈但缺乏新闻解释的资产，并附上 **Search Query**。

#### 4. 宏观与风险 (Macro & Risks)
* 基于指数表现（^DJI, ^GSPC, ^IXIC），分析整体市场情绪。
* 识别报告中提到的系统性风险（如地缘政治、通胀数据）。

#### 5. 策略展望 (Strategic Outlook)
* 基于今日的**真实**表现，给出明天的关注重点以及投资的方向和建议。
* (例如：既然 AMD 逆势大涨，是否意味着半导体见底？既然波音无视大盘下跌而上涨，是否具备独立行情？)
---

**输入数据 (Market Report):**
{report_content}
"""
    
    # 调用 Gemini API（新 API，使用异步方式）
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=analysis_prompt
            )
        )
        # 新 API 返回格式：response.text 或 response.candidates[0].content.parts[0].text
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                if len(candidate.content.parts) > 0:
                    return candidate.content.parts[0].text
        # 如果以上都不行，尝试直接转换为字符串
        return str(response)
    except Exception as e:
        return f"Gemini API 调用失败: {str(e)}"


def check_report_date(md_file_path: str) -> tuple[bool, str]:
    """
    检查 markdown 文件中的日期是否与今天一致。
    
    Args:
        md_file_path: markdown 文件路径
    
    Returns:
        (是否匹配, 文件中的日期字符串)
    """
    try:
        with open(md_file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        
        # 解析日期
        if first_line.startswith("Generated on:"):
            file_date = first_line.replace("Generated on:", "").strip()
            current_date = datetime.now().strftime("%Y-%m-%d")
            return (file_date == current_date, file_date)
        else:
            return (False, "未找到日期信息")
    except FileNotFoundError:
        return (False, "文件不存在")
    except Exception as e:
        return (False, f"读取文件错误: {str(e)}")


async def finance_info():
    async with client.run() as agent:
#         print("\nOmniBriefing Agent 已就绪")
#         await agent.interactive()

        # 自动总结
        print("\nOmniBriefing 新闻简报开始...")

        tech_tickers = ["NVDA", "TSLA", "AMD", "AAPL", "MSFT", "GOOGL"]
        industrial_tickers = ["BA", "XOM", "F"] # 卡特彼勒, 波音, 通用电气, 埃克森美孚, 福特
        
        # 大宗商品：黄金、白银、原油、铜、天然气等
        commodity_tickers = [
            "GC=F",    # COMEX 黄金期货
            "CL=F",    # WTI 原油期货
            "NG=F",    # NYMEX 天然气期货
        ]

        # 获取当前时间（用于判断周末）
        current_time = datetime.now()
        is_weekend = current_time.weekday() >= 5 # 5=Sat, 6=Sun

        user_prompt = f"""
        Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        Is Weekend: {is_weekend}

        Please generate the report based on the following watchlists:

        1. **Market Indices**: ^GSPC, ^DJI, ^IXIC
        2. **Tech Watchlist (Must Report)**: {", ".join(tech_tickers)}
        3. **Industrial Watchlist (Report only if significant change > 1.5% or major news)**: {", ".join(industrial_tickers)}
        4. **Commodities Watchlist (Must Report - Key economic indicators)**: {", ".join(commodity_tickers)}
        """

        # 2. 使用 agent.send() 让 agent 完成所有工作（包括生成报告）
        # 如果模型调用了不存在/不允许的 tool（例如把 export_final_report 写成 finance_export_final_report），
        # mcp_agent 会抛出校验异常。这里做一次自动纠错重试，避免直接退出。
        try:
            response = None
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    response = await agent.send(user_prompt)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    retryable = (
                        "tool call validation failed" in msg
                        or "Failed to call a function" in msg
                        or "failed_generation" in msg
                        or "which was not in request.tools" in msg
                    )
                    if not retryable or attempt >= 1:
                        raise
                    # 追加一次强约束提示，让模型按"真实工具名"重试
                    print(f"\n⚠ 工具调用错误，正在重试 (尝试 {attempt + 1}/2)...")
                    print(f"错误信息: {msg[:200]}")
                    user_prompt = (
                        user_prompt
                        + "- Tool args must be valid JSON format.\n"
                        + "- Retry the failed tool call with the correct name.\n"
                    )
            if response is None and last_error is not None:
                raise last_error
            
            # 3. 使用 "Daily Market Pulse" 来提取报告内容
            if "# Daily Market Pulse" in response:
                # 找到报告开始位置
                start_idx = response.find("# Daily Market Pulse")
                
                # 提取报告内容（从 "Daily Market Pulse" 开始到响应结束）
                report_content = response[start_idx:].strip()
                
                # 4. 输出分隔符和报告
                print("="*20 + " 财经简报 " + "="*20)
                print(report_content)
                
                # 保存到 Markdown 文件
                output_dir = "finance_temp_data"
                output_file = os.path.join(output_dir, "daily_briefing.md")
                # 确保目录存在
                os.makedirs(output_dir, exist_ok=True)
                
                # 获取当前日期
                current_date = datetime.now().strftime("%Y-%m-%d")
                
                # 在文件开头添加日期行
                content_with_date = f"Generated on: {current_date}\n\n{report_content}"
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(content_with_date)
                print(f"\n简报已保存到 {output_file} (日期: {current_date})")
                
                # 返回文件路径，供后续处理
                return output_file
                
            elif "##-------##" in response:
                # 如果没有找到 "Daily Market Pulse"，但找到了分隔符，输出整个响应
                print("##-------##")
                print("="*20 + " Agent 响应 " + "="*20)
                print(response)
                print("##-------##")
                print("\n⚠ 未找到报告内容，但找到了分隔符")
            else:
                # Agent 未完成或没有找到标记，输出当前响应
                print("="*20 + " Agent 响应 " + "="*20)
                print(response)
                print("\n⚠ 未找到完成标记或报告内容，可能还在处理中...")
                
        except Exception as e:
            print(f"运行出错: {e}")
            return None
    
    return None  # 如果没有成功生成报告，返回 None


async def main():
    """
    主函数：先检查今天是否已有报告，如果有则直接分析；如果没有则生成报告后再分析。
    """
    output_dir = "finance_temp_data"
    md_file_path = os.path.join(output_dir, "daily_briefing.md")
    
    # 1. 先检查今天是否已经成功生成了报告
    date_matches, file_date = check_report_date(md_file_path)
    
    if date_matches:
        print(f"✓ 检测到今天的报告已存在 (日期: {file_date})，跳过生成步骤，直接开始分析...")
    else:
        print(f"⚠ 未找到今天的报告 (文件日期: {file_date})，开始生成新报告...")
        # 2. 运行简报生成
        md_file_path = await finance_info()
        
        if md_file_path is None:
            print("简报生成失败，跳过分析步骤")
            return
        
        # 3. 再次检查文件中的日期是否与今天一致
        date_matches, file_date = check_report_date(md_file_path)
        
        if not date_matches:
            print(f"⚠ 生成的报告日期 ({file_date}) 与今天不一致，跳过分析步骤")
            return
    
    # 4. 如果日期匹配，调用分析函数
    if date_matches:
        print(f"\n✓ 报告日期 ({file_date}) 与今天一致，开始分析...")
        analysis_result = await analyze_report(md_file_path)
        print("="*20 + " 分析结果 " + "="*20)
        print(analysis_result)
        
        # 保存分析结果
        analysis_file = md_file_path.replace("daily_briefing.md", "analysis.md")
        with open(analysis_file, "w", encoding="utf-8") as f:
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(analysis_result)
        print(f"\n分析结果已保存到 {analysis_file}")


if __name__ == "__main__":
    asyncio.run(main())