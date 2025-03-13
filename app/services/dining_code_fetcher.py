import json
import redis
from bs4 import BeautifulSoup
from urllib.parse import quote
from app.shared_state import dining_code_menus
import requests
from datetime import datetime


source = "dining_code"
data = []
base_url = "https://www.diningcode.com"

def fetch_data():
    global dining_code_menus
    
    rd = redis.Redis(host='localhost', port=6379, db=0)

    # # 첫 번째 요청 (쿠키 가져오기)
    # response = session.get(base_url, headers=headers, verify=False)
    # cookies = session.cookies.get_dict()  # 동적 쿠키 저장

    # # 두 번째 요청 (검색 페이지 접근)
    # search_url = f"https://www.diningcode.com/list.dc?query={requests.utils.quote(search_query)}"
    # response = session.get(search_url, headers=headers, cookies=cookies, verify=False)
    # list_page_content = None
    # # 응답 확인
    # if response.status_code == 200:
    #     list_page_content = response.text
    # else:
    #     print(f"Error: {response.status_code}")# HTML에서 JSON 데이터 추출 (첨부 파일 ID 0 가정)

    # soup = BeautifulSoup(list_page_content, 'html.parser')
    # json_data = soup.find('script', {'type': 'application/ld+json'})
    # print(json_data.string)
    # data = json.loads(json_data.string)

    # # 추가적으로 데이터 fetching 하기


    # # URL 수정
    # for item in data.get("itemListElement", []):
    #     url = item.get("url")
    #     if url:
    #         parts = url.split(maxsplit=1)
    #         if len(parts) == 2:
    #             domain = parts[0]
    #             path_part = parts[1].strip()
    #             if '?' in path_part:
    #                 path_part_path, path_part_query = path_part.split('?', 1)
    #                 encoded_path = quote(path_part_path)
    #                 encoded_query = quote(path_part_query)  # 공백을 %20으로 인코딩
    #                 correct_path = encoded_path + '?' + encoded_query
    #             else:
    #                 encoded_path = quote(path_part)
    #                 correct_path = encoded_path
    #             item["corrected_url"] = domain + '/' + correct_path

    url = "https://im.diningcode.com/API/isearch/"

    # 검색어
    query = "정자역"
    order = "r_score"
    page = 1
    size = 20

    # 요청 헤더
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "ko,en;q=0.9,en-US;q=0.8",
        "Connection": "keep-alive",
        "Origin": "https://www.diningcode.com",
        "Referer": "https://www.diningcode.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/"
    }

    # form data 로 요청
    form_data = {
        "query": query,
        "order": order,
        "page": str(page),
        "size": str(size)
    }

    raw_result = []

    # 요청
    for i in range(1, 6):
        response = requests.post(url, headers=headers, data=form_data, verify=False)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return
        
        data = response.json()["result_data"]["poi_section"]["list"]
        raw_result.extend(data)
        form_data["page"] = str(i + 1)

    # result = []
    # # 첨부 파일로 웹 페이지 내용 가져오기
    # for idx, restaurant_info in enumerate(data.get("itemListElement", [])):
    #     # print(restaurant_info) # 데이터 확인
    #     web_page_content = requests.get(restaurant_info.get("url"), headers=headers, verify=False).text
    #     result.append(extract_restaurant_details(web_page_content))

    print(raw_result)
    
    # raw_result 는 semantic search를 위한 데이터
    from app.services.llm_handler import generate_embedding
    
    generate_embedding(source, raw_result)
        
def extract_restaurant_details(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')
    # print(soup)  # HTML 확인
    # 이름 추출: 보통 <h1 class="tit">에 있음
    name_tag = soup.find('h1', class_='tit')
    name = name_tag.text.strip() if name_tag else ""

    # 위치 추출: <li class="locat">에 여러 요소가 섞여있으므로 전체 텍스트를 가져옵니다.
    address_tag = soup.find('li', class_='locat')
    address = address_tag.get_text(separator=" ").strip() if address_tag else ""

    # 카테고리 추출: <li class="tag"> 안에 <a>태그들 (쉼표로 구분됨)
    category_tag = soup.find('li', class_='tag')
    category = category_tag.get_text(separator=",").strip() if category_tag else ""

    # 리뷰 점수 추출: <span id="lbl_review_point">에 있음
    score_tag = soup.find('strong', id='lbl_review_point')
    review_score = score_tag.text.strip() if score_tag else ""

    # 리뷰 수 추출: <span class="review_count">에 "(16명의 평가)" 같이 포함되어 있음
    review_count_tag = soup.find('span', class_='review_count')
    # 괄호 등 불필요 문자 제거: 예) "(16명의 평가)" -> 16
    review_count = ""
    if review_count_tag:
        # 숫자만 추출
        import re
        nums = re.findall(r'\d+', review_count_tag.text)
        review_count = nums[0] if nums else ""

    # 전화번호 추출: <li class="tel">
    phone_tag = soup.find('li', class_='tel')
    phone = phone_tag.text.strip() if phone_tag else ""

    # 운영시간 추출: 예시로 <div id="div_hour"> 안의 텍스트 전체를 사용
    hours_tag = soup.find('div', id='div_hour')
    hours = hours_tag.get_text(separator=" ").strip() if hours_tag else ""

    # 메뉴정보 추출:
    menu_info = []
    menu_div = soup.find('div', id='div_menu')
    if menu_div:
        menu_ul = menu_div.find('ul', class_='Restaurant_MenuList')
        if menu_ul:
            for li in menu_ul.find_all('li'):
                # 메뉴 이름는 <p class="l-txt Restaurant_MenuItem"> 안의 <span class="Restaurant_Menu"> 태그에 있을 수 있음
                item_name_tag = li.find('p', class_='l-txt Restaurant_MenuItem')
                if item_name_tag:
                    item_name_span = item_name_tag.find('span', class_='Restaurant_Menu')
                    item_name = item_name_span.text.strip() if item_name_span else item_name_tag.text.strip()
                else:
                    item_name = ""
                # 메뉴 가격은 <p class="r-txt Restaurant_MenuPrice">
                item_price_tag = li.find('p', class_='r-txt Restaurant_MenuPrice')
                item_price = item_price_tag.text.strip() if item_price_tag else ""
                if item_name:
                    menu_info.append({
                        'name': item_name,
                        'price': item_price
                    })
    return {
        'name': name,
        'address': address,
        'category': category,
        'review_score': review_score,
        'review_count': review_count,
        'phone': phone,
        'hours': hours,
        'menu': menu_info,
        "source": source,
        'date': datetime.now().strftime("%Y-%m-%d")
    }

def fetch_exact_dining_code_data(r_vid):
    url = f"https://www.diningcode.com/?rid={r_vid}"
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        return extract_restaurant_details(response.text)
    else:
        print(f"Failed to fetch data, Status code: {response.status_code}")