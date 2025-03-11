import json
import requests
from datetime import datetime, timedelta
from langchain_core.documents import Document

from app.shared_state import all_menus

def fetch_menu_data(date):
    global all_menus
    
    url = f"https://mc.skhystec.com/V3/prc/selectMenuList.prc?campus=BD&cafeteriaSeq=21&mealType=LN&ymd={date}"
    response = requests.post(url, verify=False)
    
    if response.status_code == 200:
        try:
            data = response.json()
            if data.get("RESULT") == 'N':
                data = None
            menu_data = {"date": datetime.strptime(date, '%Y%m%d').strftime("%Y-%m-%d"), "body": data}
            all_menus.append(Document(page_content=str(menu_data)))
        except json.JSONDecodeError:
            print("JSON decode error")
    else:
        print(f"Failed to fetch data, Status code: {response.status_code}")

def fetch_data():
    start_date = datetime(2025, 1, 1)
    end_date = datetime.now()
    current_date = start_date

    while current_date <= end_date:
        fetch_menu_data(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    