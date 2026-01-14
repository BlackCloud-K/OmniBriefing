import asyncio
from json import load
import os
from dotenv import load_dotenv
from datetime import datetime
from mcp_agent import FastAgent, RequestParams
import requests

load_dotenv()

def get_current_location():
    """
    通过公共 IP API 获取当前城市。
    """
    try:
        print("正在自动定位...")

        response = requests.get('http://ip-api.com/json', timeout=5)
        data = response.json()
        
        if data['status'] == 'success':
            city = data.get('city', 'unkown')
            print(f"定位成功：{city}")
            return city
        else:
            print("定位失败，使用默认城市")
            return "Beijing" # 失败兜底
    except Exception as e:
        print(f"定位出错 ({e})，使用默认城市")
        return "Beijing"

client = FastAgent("Weather Client")

@client.agent(
    name = "OmniBriefing",
    instruction= """
    你是一个专业的简报助手 (OmniBriefing Agent)。
    你的目标是根据用户的自然语言指令，调用相关工具获取数据，并生成一份结构清晰、信息准确的简报。
    简报包含当前时间（日期），地点和天气信息。
    
    当前能力：
    - 天气查询：可以精确查询全球城市的天气。
    
    回答原则：
    1. 如果用户问多个城市，请分别调用工具查询，最后汇总对比。
    2. 在简报末尾给出基于数据的实用建议（如穿衣、出行）。
    3. 【重要策略】如果天气工具返回“查无此地”或报错：
           - 请利用你的地理知识，推断该地点所属的最近的【大城市】
           - 不要问用户，直接进行重试。

    【工具调用规则】
    1. 调用工具时，必须严格遵守工具定义的参数格式以及语言。
    2. 不要臆造不存在的参数。
    3. 如果不确定参数填什么，请先询问用户，不要瞎猜。
    4. 如果不需要调用工具则不调用。
    """,
    servers = ["weather"],
    model="openai.llama-3.3-70b-versatile",
    request_params=RequestParams(
        max_iterations=10,
    )
)


async def weather_info():
    async with client.run() as agent:
#         print("\nOmniBriefing Agent 已就绪")
#         await agent.interactive()

        # 自动总结
        print("\nOmniBriefing 自动化作业开始...")

        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M") # 格式化时间
        week_day = datetime.now().strftime("%A")
        city = get_current_location()
        
        daily_task = f"地点：{city}, 日期:{current_time}, {week_day}。请开始撰写今日简报"

        # 2. 使用 agent.rend()
        try:
            response = await agent.send(daily_task)
            
            # 3. 输出或保存结果
            print("="*20 + " 每日简报 " + "="*20)
            print(response)
            
            # (可选) 保存到 Markdown 文件
            # with open("daily_briefing.md", "w", encoding="utf-8") as f:
            #     f.write(f"# 每日简报 ({os.getenv('DATE', 'Today')})\n\n{final_content}")
            #     print("\n简报已保存到 daily_briefing.md")
                
        except Exception as e:
            print(f"运行出错: {e}")

if __name__ == "__main__":
    asyncio.run(weather_info())