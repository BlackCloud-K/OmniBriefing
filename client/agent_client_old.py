import asyncio
from json import load
import os
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_agent import FastAgent, RequestParams

load_dotenv()
client = FastAgent("OmniBriefing Client")

@client.agent(
    name = "OmniBriefing",
    instruction= """
    你是一个专业的简报助手 (OmniBriefing Agent)。
    你的目标是根据用户的自然语言指令，调用相关工具获取数据，并生成一份结构清晰、信息准确的中文简报。
    
    当前能力：
    - 天气查询：可以精确查询全球城市的天气。
    
    回答原则：
    1. 如果用户问多个城市，请分别调用工具查询，最后汇总对比。
    2. 在简报末尾给出基于数据的实用建议（如穿衣、出行）。
    """,
    servers = ["weather"],
    model="google.gemini-2.5-flash",
    request_params=RequestParams(
        max_iterations=10,
    )
)
async def main():
    async with client.run() as agent:
        print("\n OmniBriefing Agent 已就绪")
        await agent.interactive()

if __name__ == "__main__":
    asyncio.run(main())