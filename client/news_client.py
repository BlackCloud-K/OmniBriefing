import asyncio
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

load_dotenv() 

brave_server = StdioServerParameters(
    command="npx", # 使用 npx 运行
    args=["-y", "@brave/brave-search-mcp-server"], # 指定包名
    env={
        "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY"),
        **os.environ
    }
)

trends_server = StdioServerParameters(
    command="npx",
    args=["-y", "mcp-trends-hub"], 
    env={**os.environ}
)

jina_server = StdioServerParameters(
    command="npx",
    args=["-y", "mcp-remote", "https://mcp.jina.ai/v1"],
    env={
        # "JINA_API_KEY": os.getenv("JINA_API_KEY") 
        **os.environ
    }
)

async def run_agent():
    # 建立连接
    async with AsyncExitStack() as stack:
        # --- 连接 Brave Search ---
        brave_transport = await stack.enter_async_context(stdio_client(brave_server))
        brave_session = await stack.enter_async_context(ClientSession(brave_transport[0], brave_transport[1]))
        await brave_session.initialize()
        print("Brave Search 已连接")

        # --- 连接 Trends Hub ---
        trends_transport = await stack.enter_async_context(stdio_client(trends_server))
        trends_session = await stack.enter_async_context(ClientSession(trends_transport[0], trends_transport[1]))
        await trends_session.initialize()
        print("Trends Hub 已连接")

        jina_transport = await stack.enter_async_context(stdio_client(jina_server))
        jina_session = await stack.enter_async_context(ClientSession(jina_transport[0], jina_transport[1]))
        await jina_session.initialize()
        print("Jina Reader 已连接")

if __name__ == "__main__":
    asyncio.run(run())