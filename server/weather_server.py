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
        location: English name of the city (such as: "Beijing", "New York", "London"), no other languages allowed.
    """
    # 检查 API_KEY 是否设置
    if not API_KEY:
        return "错误：WEATHER_API_KEY 环境变量未设置。请在 .env 文件中设置 WEATHER_API_KEY。"
    
    # 验证 location 参数
    if not location or not location.strip():
        return "错误：城市名称不能为空。"
    
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location.strip(),
        "appid": API_KEY,
        "units": "metric", 
        "lang": "zh_cn"    
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
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
            error_detail = ""
            try:
                error_data = e.response.json()
                if "message" in error_data:
                    error_detail = f" ({error_data['message']})"
            except:
                pass
            return f"查询失败：无法找到城市 '{location}' 或 API 请求错误 (状态码: {e.response.status_code}){error_detail}。"
        except httpx.TimeoutException:
            return f"查询超时：无法连接到天气服务。"
        except Exception as e:
            return f"发生未知错误: {str(e)}"


if __name__ == '__main__':
    mcp.run()