from finance_server import summarize_selected_indices, fetch_and_store_prices

res = fetch_and_store_prices(['^GSPC', '^DJI', '^IXIC', 'NVDA', 'TSLA'])

print(res)