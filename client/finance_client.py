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
    model="groq.meta-llama/llama-4-scout-17b-16e-instruct",
    request_params=RequestParams(
        max_iterations=15,
    )
)


async def finance_info():
    async with client.run() as agent:
#         print("\nOmniBriefing Agent 已就绪")
#         await agent.interactive()

        # 自动总结
        print("\nOmniBriefing 新闻简报开始...")

        tech_tickers = ["NVDA", "TSLA", "AMD", "AAPL", "MSFT", "GOOGL"]
        industrial_tickers = ["BA", "XOM", "F"] # 卡特彼勒, 波音, 通用电气, 埃克森美孚, 福特

        # 获取当前时间（用于判断周末）
        import datetime
        current_time = datetime.datetime.now()
        is_weekend = current_time.weekday() >= 5 # 5=Sat, 6=Sun

        user_prompt = f"""
        Current Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        Is Weekend: {is_weekend}

        Please generate the report based on the following watchlists:

        1. **Market Indices**: ^GSPC, ^DJI, ^IXIC
        2. **Tech Watchlist (Must Report)**: {", ".join(tech_tickers)}
        3. **Industrial Watchlist (Report only if significant change > 1.5% or major news)**: {", ".join(industrial_tickers)}
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
                    )
                    if not retryable or attempt >= 1:
                        raise
                    # 追加一次强约束提示，让模型按“真实工具名”重试
                    user_prompt = (
                        user_prompt
                        + "\n\nIMPORTANT RETRY:\n"
                        + "- You MUST retry once.\n"
                        + "- Use the EXACT tool name `export_final_report` (do NOT use `finance_export_final_report` or `finance_export_final-report`).\n"
                        + "- Use the EXACT tool name `fetch_and_store_prices` (not get_stock_data).\n"
                        + "- Use the EXACT tool name `search_news_options` (not get_market_news).\n"
                        + "- Tool args must be valid JSON.\n"
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
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(f"\n简报已保存到 {output_file}")
                
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

# 注意：不再需要直接调用 MCP server，因为会创建新的进程导致 SESSION_STATE 为空
# 现在通过 agent 调用 export_final_report，使用同一个 server 进程的 SESSION_STATE

if __name__ == "__main__":
    asyncio.run(finance_info())