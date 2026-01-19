import asyncio
import os
import json
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# 配置你的 Server (这里以 Brave 为例，你可以换成任何 Server)
server_params = StdioServerParameters(
    command="npx",
    args = [
        "-y",
        "mcp-trends-hub@1.6.2"
      ]
)

async def inspect_tools():
    print("⏳ 正在连接 Server...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 1. 获取工具列表
            print("🔍 正在获取工具列表 (list_tools)...")
            response = await session.list_tools()
            
            tools = response.tools
            print(f"\n✅ 成功发现 {len(tools)} 个工具：\n" + "="*40)

            # 2. 遍历并打印详细信息
            for i, tool in enumerate(tools, 1):
                print(f"\n🔧 工具 #{i}: {tool.name}")
                print(f"DESC: {tool.description}")
                
                # 3. 漂亮地打印参数结构 (Schema)
                if tool.inputSchema:
                    print("ARGS: ", end="")
                    print(json.dumps(tool.inputSchema, indent=2, ensure_ascii=False))
                
                print("-" * 40)

if __name__ == "__main__":
    asyncio.run(inspect_tools())