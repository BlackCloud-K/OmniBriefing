import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv
from mcp_agent import FastAgent, RequestParams

load_dotenv() 

# 模型配置：优先使用 deepseek v3，失败时自动切换到 llama3
# 注意：需要设置 OPENROUTER_API_KEY 环境变量（在 .env 文件中）
# 获取方式：访问 https://openrouter.ai，注册账号，在 Dashboard 创建 API Key
PRIMARY_MODEL = "openrouter.deepseek/deepseek-v3.2"
FALLBACK_MODEL = "openrouter.meta-llama/llama-3.3-70b-instruct:free"

# 创建主 agent（使用 deepseek v3）
client_primary = FastAgent("News Agent - Primary")

@client_primary.agent(
    name = "News Agent",
    instruction="""
        You are a **Strategic Intelligence Analyst** specializing in Global Tech & Geopolitics. 
        Your goal is not just to summarize news, but to **connect the dots**, analyze implications, and provide a multi-perspective briefing in Chinese.

        **CORE DIRECTIVES:**
        1.  **Source Strategy (Hybrid)**: 
            - **Primary (80%)**: Prioritize English platforms (US/UK/Global) for primary facts, data, and depth.
            - **Secondary (20%)**: You **MUST** include at least one trending topic or perspective from **Chinese platforms** to ensure a balanced view.
        2.  **Analysis over Summary**: Do not just list facts. Explain *why* it matters and *how* Event A relates to Event B.
        3.  **Token Efficiency**: Limit yourself to **6-8 high-value searches** total.

        **WORKFLOW:**

        **STEP 1: Trend Acquisition (Dual-Core)**
        - Use `trends` tools to fetch top stories from **both** US/Global sources (e.g., BBC, NYT, HackerNews) **AND** Chinese sources (e.g., Weibo, Zhihu).
        - Select the **Top 5** most significant topics combined.
        - **Selection Criteria**: Prioritize topics where China and the West have **conflicting views** or **strong economic ties**.

        **STEP 2: Investigative Search**
        - For each selected topic, perform targeted searches.
        - **Mandatory**: For major geopolitical or tech events, search in **English** for the main scoop, and search in **Chinese** (if applicable) for the local reaction or impact.
        - *Example*: If searching "Nvidia export ban", check US official statements (English) AND Chinese market reaction (Chinese).

        **STEP 3: Synthesis & Forecasting (The "Brain" Step)**
        - **Connect**: How does this tech news affect politics? How does this US policy affect Chinese markets?
        - **Forecast**: What is the immediate next step? What should the user watch for next week?

        **STRICT OUTPUT FORMAT (Markdown):**

        You must output the report in the following structure. Use Chinese for the content.

        # 新闻简报

        ## 核心动态 (Key Intelligence)
        *(Select 3-4 most critical stories. Mix of Global & Chinese topics)*

        ### 1. [新闻标题]
        - **事实锚点 (Global Fact)**: [Based on English sources: Key data, official announcement]
        - **多方视角**
        - **深度分析 (Insight)**: [Your analysis: Why this matters. Connection to other events.]

        ### 2. [新闻标题]
        ...

        ## 链式反应与洞察 (Connections & Analysis)
        *(Don't just summarize. Analyze the hidden links between the stories above)*
        - **事件关联**: [Example: "News #1's chip ban directly caused the stock drop in News #3..."]
        - **趋势研判**: [Your professional judgment on the macro trend]

        ## 关注雷达 (Future Radar)
        *(Tell the user what to look out for in the next 7 days)*
        - **[Short Term]**: [e.g., "Watch for the Fed meeting minutes on Wednesday..."]
        - **[Long Term]**: [e.g., "Pay attention to DeepSeek's API stability..."]

        ---
    """,
    servers = ["trends"],  # 移除 brave（频繁报错）和 jina，trends 可以提供热门话题和搜索功能
    model=PRIMARY_MODEL,  # 主模型：DeepSeek V3
    request_params=RequestParams(
        max_iterations=10,
    )
)
async def summarize_news_primary():
    """使用主模型（DeepSeek V3）生成简报"""
    pass

# 创建备用 agent（使用 llama3）
client_fallback = FastAgent("News Agent - Fallback")

@client_fallback.agent(
    name = "News Agent",
    instruction="""
        You are a **Strategic Intelligence Analyst** specializing in Global Tech & Geopolitics. 
        Your goal is not just to summarize news, but to **connect the dots**, analyze implications, and provide a multi-perspective briefing in Chinese.

        **CORE DIRECTIVES:**
        1.  **Source Strategy (Hybrid)**: 
            - **Primary (80%)**: Prioritize English platforms (US/UK/Global) for primary facts, data, and depth.
            - **Secondary (20%)**: You **MUST** include at least one trending topic or perspective from **Chinese platforms** to ensure a balanced view.
        2.  **Analysis over Summary**: Do not just list facts. Explain *why* it matters and *how* Event A relates to Event B.
        3.  **Token Efficiency**: Limit yourself to **6-8 high-value searches** total.

        **WORKFLOW:**

        **STEP 1: Trend Acquisition (Dual-Core)**
        - Use `trends` tools to fetch top stories from **both** US/Global sources (e.g., TechCrunch, NYT, HackerNews) **AND** Chinese sources (e.g., Weibo, 36Kr).
        - Select the **Top 5** most significant topics combined.
        - **Selection Criteria**: Prioritize topics where China and the West have **conflicting views** or **strong economic ties**.

        **STEP 2: Investigative Search**
        - For each selected topic, perform targeted searches.
        - **Mandatory**: For major geopolitical or tech events, search in **English** for the main scoop, and search in **Chinese** (if applicable) for the local reaction or impact.
        - *Example*: If searching "Nvidia export ban", check US official statements (English) AND Chinese market reaction (Chinese).

        **STEP 3: Synthesis & Forecasting (The "Brain" Step)**
        - **Connect**: How does this tech news affect politics? How does this US policy affect Chinese markets?
        - **Forecast**: What is the immediate next step? What should the user watch for next week?

        **STRICT OUTPUT FORMAT (Markdown):**

        You must output the report in the following structure. Use Chinese for the content.

        # 新闻简报

        ## 核心动态 (Key Intelligence)
        *(Select 3-4 most critical stories. Mix of Global & Chinese topics)*

        ### 1. [新闻标题]
        - **事实锚点 (Global Fact)**: [Based on English sources: Key data, official announcement]
        - **多方视角**
        - **深度分析 (Insight)**: [Your analysis: Why this matters. Connection to other events.]

        ### 2. [新闻标题]
        ...

        ## 链式反应与洞察 (Connections & Analysis)
        *(Don't just summarize. Analyze the hidden links between the stories above)*
        - **事件关联**: [Example: "News #1's chip ban directly caused the stock drop in News #3..."]
        - **趋势研判**: [Your professional judgment on the macro trend]

        ## 关注雷达 (Future Radar)
        *(Tell the user what to look out for in the next 7 days)*
        - **[Short Term]**: [e.g., "Watch for the Fed meeting minutes on Wednesday..."]
        - **[Long Term]**: [e.g., "Pay attention to DeepSeek's API stability..."]

        ---
    """,
    servers = ["trends"], 
    model=FALLBACK_MODEL,  
    request_params=RequestParams(
        max_iterations=10,
    )
)
async def summarize_news_fallback():
    """使用备用模型（Llama3）生成简报"""
    pass

async def summarize_news():
    current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    week_day = datetime.now().strftime("%A")
    interest_fields = ["Technology", "Politics (China, US, and hotspot regions)", "International situations and regional conflicts", "General hot topics"]
    initial_prompt = f"今天是{current_time}, {week_day}。请开始今天的新闻简报撰写。重点关注以下领域：{', '.join(interest_fields)}。**"
    
    response = None
    accumulated_context = ""  # 保存已获取的上下文信息
    last_partial_response = ""  # 保存最后一次的部分响应
    
    # 优先尝试使用 DeepSeek V3
    try:
        print(f"尝试使用模型: {PRIMARY_MODEL}")
        async with client_primary.run() as agent:
            # 使用迭代方式执行，以便捕获中间状态
            try:
                response = await agent.send(initial_prompt)
                print(f"✓ 成功使用模型: {PRIMARY_MODEL}")
            except Exception as mid_execution_error:
                # 运行中断时的处理
                error_msg = str(mid_execution_error).lower()
                is_model_error = any(keyword in error_msg for keyword in [
                    "model", "unavailable", "rate limit", "quota", "not found", 
                    "invalid", "timeout", "connection", "api key", "401", "403",
                    "interrupted", "aborted", "cancelled"
                ])
                
                if is_model_error:
                    print(f"模型 {PRIMARY_MODEL} 在执行过程中中断: {str(mid_execution_error)}")
                    print(f"切换到备用模型: {FALLBACK_MODEL} 继续执行...")
                    
                    # 构建续接提示，包含原始任务和已执行的信息
                    continuation_prompt = f"""{initial_prompt}

                    **重要：之前的执行被中断。请继续完成新闻简报的撰写任务。如果已经获取了一些新闻信息，请基于已有信息继续完成简报。如果还没有开始，请从头开始执行完整的工作流程。"""
                    
                    # 切换到备用模型继续执行
                    async with client_fallback.run() as fallback_agent:
                        response = await fallback_agent.send(continuation_prompt)
                        print(f"✓ 备用模型 {FALLBACK_MODEL} 成功接续完成任务")
                else:
                    # 非模型错误，直接抛出
                    raise
    except Exception as e:
        error_msg = str(e).lower()
        # 检查是否是模型相关的错误（启动时失败）
        is_model_error = any(keyword in error_msg for keyword in [
            "model", "unavailable", "rate limit", "quota", "not found", 
            "invalid", "timeout", "connection", "api key", "401", "403"
        ])
        
        if is_model_error:
            print(f"✗ 模型 {PRIMARY_MODEL} 启动失败: {str(e)}")
            print(f"切换到备用模型: {FALLBACK_MODEL}")
            # 切换到备用模型从头开始
            try:
                async with client_fallback.run() as agent:
                    response = await agent.send(initial_prompt)
                    print(f"✓ 成功使用备用模型: {FALLBACK_MODEL}")
            except Exception as fallback_error:
                raise Exception(f"主模型和备用模型都失败。主模型错误: {e}, 备用模型错误: {fallback_error}")
        else:
            # 其他错误（如工具调用错误），直接抛出
            raise
    
    # 处理响应
    if response and "# 新闻简报" in response:
            # 找到报告开始位置
            start_idx = response.find("# 新闻简报")
            
            # 提取报告内容（从 "# 新闻简报" 开始到响应结束）
            report_content = response[start_idx:].strip()
            
            # 4. 输出分隔符和报告
            print("="*40)
            print(report_content)
            
            # 保存到 Markdown 文件
            current_date = datetime.now().strftime("%Y-%m-%d")
            output_dir = "news_temp_data"
            output_file = os.path.join(output_dir, "news_briefing.md")
            # 确保目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 在文件开头添加日期行
            content_with_date = f"Generated on: {current_date}\n\n{report_content}"
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content_with_date)
            print(f"\n简报已保存到 {output_file} (日期: {current_date})")
    
            print("新闻报告生成完成，任务结束")


if __name__ == "__main__":
    asyncio.run(summarize_news())