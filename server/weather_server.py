from mcp.server.fastmcp import FastMCP
import httpx
import os 
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")

mcp = FastMCP("weather", dependencies=['httpx'])

@mcp.tool()
async def getWeather(location: str) -> str:
    """
    Get the weather information for the given location(city)

    Args:
        location: English name of the city (such as: "Beijing", "New York", "London")
    """

    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": API_KEY,
        "units": "metric", 
        "lang": "zh_cn"    
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # 解析数据
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            
            # 返回一段描述
            return f"{location} 当前天气：{weather_desc}，温度：{temp}°C，湿度：{humidity}%。"
            
        except httpx.HTTPStatusError as e:
            return f"查询失败：无法找到城市 '{location}' 或 API 请求错误 (状态码: {e.response.status_code})。"
        except Exception as e:
            return f"发生未知错误: {str(e)}"


if __name__ == '__main__':
    mcp.run()