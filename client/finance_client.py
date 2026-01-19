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
        You are a financial data curator. Execute tasks immediately. Do NOT plan, describe, or explain - just execute.

        **STEP 1: Get Prices**
        - Call `fetch_and_store_prices(tickers=[ALL_TICKERS], prepost=True)` with all tickers from the user's watchlist.

        **STEP 2: Select Stocks by Volatility**
        - Review price changes from Step 1.
        - Select 5-8 stocks with |change_percent| > 1% (highest volatility first).
        - These are the ONLY stocks you will fetch news for.

        **STEP 3: Get News for Selected Stocks**
        - Call `search_news_options(tickers=[SELECTED_STOCKS], limit=3)` ONLY for stocks from Step 2.
        - Review the news menu.
        - Select news indices (prioritize important keywords: Earnings, Guidance, Acquisition, Layoffs, CEO, FDA, Lawsuit).
        - Call `summarize_selected_indices(indices=[...], focus_instruction="Financial news summary")` ONCE with all selected indices.

        **STEP 4: Quality Control**
        - Review summaries. Remove low-quality or duplicate news using `remove_news_summaries(indices=[...])`.
        - **CRITICAL**: Each stock from Step 2 MUST keep at least 1 news summary. Never delete all news for a selected stock.

        **STEP 5: Generate Report**
        - Call `export_final_report()` ONCE.
        - After receiving the result, respond with "Task completed." and STOP.
        - Do NOT call any other tools after this.

        **REMEMBER: Execute tools immediately. Do NOT plan or describe - just execute.**
    """,
    servers = ["finance"],
    # 使用 Groq 模型，格式：groq.model-name
    # 注意：需要确保 OPENAI_API_KEY 环境变量设置为 Groq API key（与 server 端一致）
    # model="groq.llama-3.3-70b-versatile",
    model="groq.meta-llama/llama-4-scout-17b-16e-instruct",
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
        你是一位拥有 20 年经验的华尔街**首席交易员 (Chief Trader)**. 你的陈述需要专业但是易于任何投资背景的人理解。
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
        Please generate the report based on the following watchlists. 

        Watchlists:
        1. Market Indices: ^GSPC, ^DJI, ^IXIC
        2. Tech: {", ".join(tech_tickers)}
        3. Industrial: {", ".join(industrial_tickers)}
        4. Commodities: {", ".join(commodity_tickers)}

        Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        Is Weekend: {is_weekend}

        Start executing Step 1 now.
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
                    
                    # 详细的错误信息输出
                    print("\n" + "="*60)
                    print("工具调用错误详情")
                    print("="*60)
                    print(f"错误类型: {type(e).__name__}")
                    print(f"错误消息: {msg}")
                    
                    # 尝试从异常对象中提取 failed_generation 和工具调用信息
                    failed_generation = None
                    tool_call_info = None
                    
                    # 检查异常对象的 body 属性（Groq API 错误通常在这里）
                    if hasattr(e, 'body') and isinstance(e.body, dict):
                        failed_generation = e.body.get('failed_generation')
                        print(f"\n错误代码: {e.body.get('code', 'N/A')}")
                        print(f"错误类型: {e.body.get('type', 'N/A')}")
                        
                        # 尝试从 failed_generation 中提取工具调用信息
                        if failed_generation:
                            print(f"\n{'='*60}")
                            print("Agent 尝试生成的内容 (failed_generation):")
                            print("="*60)
                            print(failed_generation[:1000])  # 限制长度避免输出过长
                            if len(failed_generation) > 1000:
                                print(f"\n... (已截断，完整内容见下方)")
                            print("="*60)
                    
                    # 检查异常对象的其他属性
                    if hasattr(e, '__dict__'):
                        for key, value in e.__dict__.items():
                            if key == 'body' and isinstance(value, dict):
                                # body 已经在上面处理了
                                continue
                            elif key not in ['request', 'message']:  # 跳过一些不重要的属性
                                if isinstance(value, (str, int, float, bool, type(None))):
                                    print(f"{key}: {value}")
                    
                    # 尝试从错误消息和 failed_generation 中提取工具名称
                    import re
                    tool_patterns = [
                        r"`([a-z_]+)`",  # 反引号中的工具名
                        r"call\s+([a-z_]+)",  # "call tool_name"
                        r"tool\s+['\"]([^'\"]+)['\"]",  # tool "name"
                        r"function\s+['\"]([^'\"]+)['\"]",  # function "name"
                        r"attempted to call tool '([^']+)'",  # attempted to call tool 'name'
                    ]
                    
                    found_tools = set()
                    search_text = msg
                    if failed_generation:
                        search_text += "\n" + failed_generation
                    
                    for pattern in tool_patterns:
                        matches = re.findall(pattern, search_text, re.IGNORECASE)
                        for match in matches:
                            if match and match not in ['call', 'tool', 'function', 'use']:
                                found_tools.add(match)
                    
                    if found_tools:
                        print(f"\n检测到可能的问题工具调用:")
                        for tool in found_tools:
                            is_valid = tool in ['fetch_and_store_prices', 'search_news_options', 
                                               'summarize_selected_indices', 'remove_news_summaries', 
                                               'export_final_report']
                            status = "✓ 正确" if is_valid else "✗ 错误（工具不存在）"
                            print(f"  - {tool}: {status}")
                    
                    print("\n可用工具列表:")
                    print("  - fetch_and_store_prices")
                    print("  - search_news_options")
                    print("  - summarize_selected_indices")
                    print("  - remove_news_summaries")
                    print("  - export_final_report")
                    
                    # 保存完整的 failed_generation 到文件
                    if failed_generation:
                        debug_file = os.path.join("finance_temp_data", "debug_failed_generation.txt")
                        os.makedirs("finance_temp_data", exist_ok=True)
                        with open(debug_file, "w", encoding="utf-8") as f:
                            f.write(f"Error: {msg}\n")
                            f.write(f"Error Type: {type(e).__name__}\n")
                            f.write(f"\n{'='*60}\n")
                            f.write("Failed Generation (Agent's attempted output):\n")
                            f.write(f"{'='*60}\n")
                            f.write(failed_generation)
                        print(f"\n完整 failed_generation 已保存到: {debug_file}")
                    
                    print("="*60 + "\n")
                    
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
                    user_prompt = (
                        user_prompt
                        + "\n\nCRITICAL: Tool call failed. Use EXACT tool names:\n"
                        + "- fetch_and_store_prices\n"
                        + "- search_news_options\n"
                        + "- summarize_selected_indices\n"
                        + "- remove_news_summaries\n"
                        + "- export_final_report\n"
                        + "Do NOT add prefixes like 'finance_' or change the names.\n"
                    )
            
            if response is None and last_error is not None:
                raise last_error
            
            # 保存完整的 agent 响应用于调试（可选，通过环境变量控制）
            debug_mode = os.getenv("DEBUG_AGENT_RESPONSE", "false").lower() == "true"
            if debug_mode:
                debug_file = os.path.join("finance_temp_data", "debug_agent_response.txt")
                os.makedirs("finance_temp_data", exist_ok=True)
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"{'='*60}\n")
                    f.write("Complete Agent Response:\n")
                    f.write(f"{'='*60}\n")
                    f.write(response)
                print(f"完整 Agent 响应已保存到: {debug_file}")
            
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
                
                # 报告已成功生成，即使 agent 后续尝试调用其他工具，也忽略错误
                # 因为任务已经完成
                print("✓ 报告生成完成，任务结束")
                
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
            print("\n" + "="*60)
            print("最终错误 - 任务失败")
            print("="*60)
            print(f"错误类型: {type(e).__name__}")
            print(f"错误消息: {str(e)}")
            
            # 打印完整堆栈
            import traceback
            print("\n完整错误堆栈:")
            traceback.print_exc()
            
            # 如果有响应，尝试保存用于调试
            try:
                if 'response' in locals() and response:
                    debug_file = os.path.join("finance_temp_data", "debug_last_response.txt")
                    os.makedirs("finance_temp_data", exist_ok=True)
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Error Type: {type(e).__name__}\n")
                        f.write(f"\n{'='*60}\n")
                        f.write("Agent Response:\n")
                        f.write(f"{'='*60}\n")
                        f.write(response)
                    print(f"\nAgent 响应已保存到: {debug_file}")
            except:
                pass
            
            print("="*60 + "\n")
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