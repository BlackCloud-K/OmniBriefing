import yfinance as yf
import json

def debug_news_structure():
    ticker = "NVDA"
    print(f"🔍 Fetching raw news for {ticker}...")
    
    stock = yf.Ticker(ticker)
    news_list = stock.news
    
    if not news_list:
        print("❌ No news found.")
        return

    print(f"✅ Found {len(news_list)} items.")
    
    # === 关键：打印第一条完整的原始数据，看看 Key 到底长什么样 ===
    first_item = news_list[0]
    print("\n--- RAW ITEM STRUCTURE (Copy this) ---")
    print(json.dumps(first_item, indent=2))
    print("--------------------------------------")

if __name__ == "__main__":
    debug_news_structure()