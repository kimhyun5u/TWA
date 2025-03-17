
import redis
import json
import os
import re

from datetime import datetime, timedelta
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command

from app.config import settings

from app.shared_state import dining_code_menus

from app.services.hyteria import fetch_menu_data as fetch_hyteria_data
from app.services.dining_code_fetcher import fetch_exact_dining_code_data

from app.structures.date_output import DateOutput
from app.structures.hyteria_menu_output import HyteriaMenuOutputList
from app.structures.dining_code_restaurant_output import DiningCodeRestaurantOutputList

# Initialize the retriever
vectorstores = {}

embeddings = AzureOpenAIEmbeddings(
    model=settings.aoai_deploy_embed_3_large,
    openai_api_version="2024-02-01",
    api_key=settings.aoai_api_key,
    azure_endpoint=settings.aoai_endpoint
)

# o3
o3_llm = AzureChatOpenAI(
    openai_api_version=settings.aoai_o3_mini_version,
    azure_deployment=settings.aoai_o3_mini_deployment_name,
    # temperature=0.0,
    api_key=settings.aoai_o3_mini_api_key,
    azure_endpoint=settings.aoai_o3_mini_endpoint
)

# 4o mini
llm = AzureChatOpenAI(
    openai_api_version=settings.aoai_deploy_gpt4o_version,
    azure_deployment=settings.aoai_deploy_gpt4o_mini,
    temperature=0.0,
    api_key=settings.aoai_api_key,
    azure_endpoint=settings.aoai_endpoint
)

sources = ["hyteria", "dining_code"]

for source in sources:
    path = f"{source}_faiss_index"
    if os.path.exists(path) and os.path.exists(f"{path}/index.pkl"):
        # Vector store already exists, load it
        print("Loading existing vector store from disk...")
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        vectorstores[source] = vectorstore
    else:
        # Vector store doesn't exist yet
        print("No existing vector store found. Will create when needed.")

def generate_embedding(source, menus):
    menus = [Document(page_content=str(menu), metadata={"source": "https://www.diningcode.com"}) for menu in menus]

    print("Embeddings created:", embeddings)
    # Check if previously saved vectorstore exists
    path = f"{source}_faiss_index"
    if os.path.exists(path) and os.path.exists(f"{path}/index.pkl"):
        print("Loading existing vectorstore from disk...")
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        # update vectorstore with new documents
        print("Updating vectorstore with new documents...")
        vectorstore.aadd_documents(menus)
        print("Vectorstore updated")
    else:
        # Create new vectorstore if not exists
        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(documents=menus, embedding=embeddings)
        # Save vectorstore to disk for future use
        vectorstore.save_local(path)
        print("Vectorstore saved to disk")
    
    print("Vectorstore created/loaded:", vectorstore)
    vectorstores[source] = vectorstore

def generate_embeddings():
    for source, menus in [("dining_code", dining_code_menus)]:
        generate_embedding(source, menus)

# Create tools
@tool
def get_today_date():
    """오늘의 날짜를 출력한다."""
    return datetime.now().strftime("%Y-%m-%d")

@tool
def get_previous_date(d: Annotated[int, "며칠 전의 값을 가져온다."]):
    """이전 날짜를 계산해 출력한다."""
    return (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")

@tool
def get_next_date(d: Annotated[int, "며칠 후의 값을 가져온다."]):
    """이후 날짜를 계산해 출력한다."""
    return (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d")

@tool
def get_hyteria_menus(date: Annotated[str, "질의 내용 중 하이테리아(구내식당, hyteria) VectorDB 에서 유사도 검색이 필요할 것들에 대한 질의"]):
    """메뉴에 대해서 검색하고 실제 있는 값인지 확인한다."""
    return fetch_hyteria_data(date)

@tool
def get_dining_code_menus(message: Annotated[str, "질의 내용 중 다이닝코드(dining_code) VectorDB 에서 유사도 검색이 필요할 것들에 대한 질의"]):
    """메뉴에 대해서 검색하고 실제 있는 값인지 확인한다."""
    if "dining_code" not in vectorstores:
        return "No retriever available. Please try again later."
    
    r = vectorstores["dining_code"].similarity_search(message, k = 10)
    
    return r

@tool
def get_exact_dining_code_data(v_rid: Annotated[str, "질의 내용 중 다이닝코드(dining_code) 사이트에서 정확한 값이 필요한 경우 restaurant_id 값을 활용해 제공한다."]):
    """매장에 대해서 검색하고 상세 정보를 확인한다."""
    print(v_rid)
    return fetch_exact_dining_code_data(v_rid)

# Create Agent Supervisor
members = ["calander", "hyteria_menu_retriever", "dining_code_menu_retriever", "menu_recommander"]

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    "If user won't ask about menu or restaurant, You MUST not Answer. Just response with FINISH."
    "You Should use calander worker if user'll ask about date relate. "
    "구내식당 = hyteria, [다이닝코드, 외부] = dining_code"
    "If date related question is asked, You Should use calander worker before other workers. "
    f"If requests doesn't include any specific menu or restaurants, You Should use all of menu retriever worker."
    "You Should use hyteria_menu_retriever worker if user'll ask about hyteria menu. "
    "You Should use dining_code_menu_retriever worker if user'll ask about dining_code menu or eat outside. "
    "You Should use menu_recommander worker if user'll ask about recommand menu. And You should use menu_recommander worker after hyteria_menu_retriever and dining_code_menu_retriever. "
    "If no more workers are needed, respond with FINISH."
    "When finished, respond with FINISH."
)

class State(MessagesState):
    next: str

def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

# Construct Graph
calander_agent = create_react_agent(
    model=llm, tools=[get_next_date, get_previous_date, get_today_date]
    , prompt="You are a calander master. You can retrieve today date, previous date and next date. Do not recommend menu. 답변에 대한 추가적인 의견을 제공하지 말고 날짜만 답변에 포함해주세요. Don't refer the history. Don't recommand"
    , response_format=DateOutput
)

hyteria_menu_retriever_agent = create_react_agent(
    llm, tools=[get_hyteria_menus]
    , prompt="You are menu retriever. You can check all menu on following date with Menu VectorDB. Do Not Math. 정확히 주어진 날짜에 대한 음식만 찾아. Do not recommend menu. 답변에 추가적인 의견을 제공하지 말고 메뉴 정보들만 제공해주세요. 정보를 변형하지말고 정확하게 전달해주세요."
    , response_format=HyteriaMenuOutputList
)

dining_code_menus_retriever_agent = create_react_agent(
    llm, tools=[get_dining_code_menus, get_exact_dining_code_data]
    , prompt="You are restaurant retriever. You can check all menu on following date with Restaurant VectorDB. Do Not Math. Do not recommend restaurant. 답변에 추가적인 의견을 제공하지 말고 레스토랑 정보들만 제공해주세요. 사용자의 요청의 적합한 레스토랑만 가져오세요. 모든 레스토랑을 원하는 경우 모든 레스토랑을 가져와주세요. 정보를 변형하지말고 정확하게 전달해주세요. 다른 에이전트에서 전달 받은 값과 상관없이 dining_code vector DB 의 값을 조회해주세요. Don't recommand."
    , response_format=DiningCodeRestaurantOutputList
)

menu_recommander_agent = create_react_agent(
    llm, tools=[get_exact_dining_code_data], prompt="You are menu recommander. 다른 agent에 의해 전달받은 값이 없으면 답을 줄 수 없습니다. 사용자가 원하는 메뉴를 주어진 메뉴들 중에 골라주세요. 만약 면요리에 대해서 물어본다면 국수, 라면, 파스타 등에 대한 정보를 찾아주세요. 또한, 답변은 상세하게 진행해주세요. 하이테리아는 hyteria 입니다. 다이닝코드는 dining_code 입니다. 구내식당(hyteria)와 외부식당(dining_code)의 메뉴를 모두 고려해주세요. 답변에 대한 추가적인 의견을 제공하지 말고 메뉴 정보들만 제공해주세요. 리스트에 restaurant_id 가 있으면 그에 대한 정보를 가져와주세요. 정보를 변형하지말고 정확하게 전달해주세요. 출처가 다이닝코드인 값은 반드시 툴을 활용해 상세 정보를 조회하세요. 질의에 restaurant_id 값이 존재한다면 상세 조회 툴을 사용하세요. 알 수 없는 내용에 대해서는 모른다고 대답하세요."
)

def calander_node(state: State) -> Command[Literal["supervisor"]]:
    
    result = calander_agent.invoke(state)

    # Extract the response content
    # content = result["messages"][-1].content
    
    # Process the content to extract date information
    # You can use an LLM call here to parse the content into structured data if needed
    # try:
    #     # Using llm to parse the calendar content into a structured format
    #     structured_content = llm.with_structured_output(DateOutput).invoke(
    #         f"Extract the date information from this text: {content}"
    #     )
        
    #     # Create a formatted response that includes the structured date info
    #     formatted_response = structured_content.date
    #     if hasattr(structured_content, 'description') and structured_content.description:
    #         formatted_response += f", Description: {structured_content.description}"
    # except Exception as e:
    #     print(f"Error parsing calendar response: {e}")
    #     formatted_response = content  # Fallback to original content
    
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["structured_response"].date, name="calander")
            ]
        },
        goto="supervisor"
    )

def hyteria_menu_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    if "messages" not in state or not state["messages"]:
        return Command(goto="calander")
    date = state["messages"][-1].content
    # validate date with regex if it is not in the correct format(YYYY-MM-DD) go to calander node
    if not re.match(r"\d{4}-\d{2}-\d{2}", date):
        return Command(goto="calander")
    
    content = get_hyteria_menus(date)
    # # result = hyteria_menu_retriever_agent.invoke(state)

    # # Extract the response content
    # # content = result["messages"][-1].content
    
    # print("content: ", content)

    # Process the content to extract date information
    # You can use an LLM call here to parse the content into structured data if needed
    try:
        # Using llm to parse the calendar content into a structured format
        processed_content = llm.invoke(
            f"다음 리스트를 ```markdown``` 같은 것을 넣지말고 마크다운문법으로 예쁘게 만들어줘(메뉴 이미지의 baseurl은 '20250313'_'BD'_'2'_'4'_LN_1_20250313111249_0.jpg 이면 https://mc.skhystec.com/nsf/menuImage/'20250313'/'BD'/'2'/'4'/ 이야): {content}"
        )
    except Exception as e:
        print(f"Error parsing data response: {e}")
        formatted_response = content  # Fallback to original content
    
    formatted_response = processed_content.content
    
    # print(formatted_response)
    return Command(
        update={
            "messages": [
                HumanMessage(content=str(formatted_response), name="hyteria_menu_retriever"),
            ]
        },
        goto="supervisor"
    )

def dining_code_menu_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    result = dining_code_menus_retriever_agent.invoke(state)

    # Extract the response content
    content = result["structured_response"].restaurants
    
    # Process the content to extract date information
    # You can use an LLM call here to parse the content into structured data if needed
    # formatted_response = llm.invoke(
    #         f"다음 리스트를 ```markdown``` 같은 것을 넣지말고 마크다운문법으로 예쁘게 만들어줘(모든 정보들이 포함되어야해 특히! restaurant_id 빼먹지마)): {content}"
    #     ).content
    
    

    return Command(
        update={
            "messages": [
                HumanMessage(content=str(content), name="dining_code_menu_retriever"),
            ]
        },
        goto="supervisor"
    )

def menu_recommander_node(state: State) -> Command[Literal["supervisor"]]:
    result = menu_recommander_agent.invoke(state)

    content = result["messages"][-1].content

    # Extract the response content
    # Process the content to extract date information
    # You can use an LLM call here to parse the content into structured data if needed
    try:
        # Using llm to parse the calendar content into a structured format
        formatted_response = llm.invoke(
            f"다음 리스트를 ```markdown``` 같은 것을 넣지말고 스마트 브레비티 기법을 사용해 마크다운문법으로 예쁘게 만들어줘 식당 메뉴가 있다면 제공하고 이미지가 있다면 보여줘(추천하는 이유들을 상세히 설명해줘 또한, 사용자의 질문에 적절한 추천인지도 생각해봐.): {content}"
        ).content

    except Exception as e:
        print(f"Error parsing data response: {e}")
        formatted_response = content  # Fallback to original content

    return Command(
        update={
            "messages": [
                HumanMessage(content=formatted_response, name="menu_recommander")
            ]
        },
        goto="supervisor"
    )

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("calander", calander_node)
builder.add_node("hyteria_menu_retriever", hyteria_menu_retriever_node)
builder.add_node("dining_code_menu_retriever", dining_code_menu_retriever_node)
builder.add_node("menu_recommander", menu_recommander_node)
graph = builder.compile()

def generate_prompt(messages: str, user_id: str):
    rd = redis.Redis(host='localhost', port=6379, db=0)

    # Get the data from Redis and deserialize it
    user_taste_json = rd.get(user_id)
    if user_taste_json:
        user_taste = user_taste_json.decode("utf-8")
    else:
        user_taste = ""

    # print("user_taste:", user_taste)

    s_prompt = SystemMessagePromptTemplate.from_template(
        f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Today is {datetime.now().strftime('%Y-%m-%d')}.
        Think about menus of {datetime.now().strftime('%Y-%m-%d')} and answer the following question.
        And Should think about the menu's date.
        What's date of today? And Think about the date in Question.
        만약 메뉴 추천을 원한다면 {datetime.now().strftime('%Y-%m-%d')}의 메뉴들을 찾아보고, 그 중에서 적합한 메뉴를 추천해주세요.
        단순 메뉴를 나열하는 것이 아닌 추천을 요청받으면 반드시 추천해주세요.
        ```markdown``` 같은 것을 넣지말고 마크다운문법으로 최종 결과를 출력해주세요.

        Answer in Korean.
        """)
    u_prompt = HumanMessagePromptTemplate.from_template(f"""
        #사용자의 취향은 다음과 같습니다. 참고해서 메뉴 추천을 진행해주세요.
        {user_taste}

        #Question:
        {messages}

        #Answer:"""
    )

    chat_prompt = ChatPromptTemplate.from_messages([s_prompt.format(), u_prompt.format()])
    
    full_result = [{"role": "user", "content": chat_prompt.format()}]
    result = []
    for c in graph.stream({"messages": [ ("user", chat_prompt.format())]}):
        # chunk에서 messages의 마지막 항목을 추출 (응답이 여기에 있다고 가정)
        print(c)
        if "hyteria_menu_retriever" in c and "messages" in c["hyteria_menu_retriever"]:
            result.append({"role": "hyteria_menu_retriever","content": c["hyteria_menu_retriever"]["messages"][-1].content})
        if "dining_code_menu_retriever" in c:
            result.append({"role": "dining_code_menu_retriever","content": c["dining_code_menu_retriever"]["messages"][-1].content})
        if "menu_recommander" in c:
            result.append({"role": "menu_recommander","content": c["menu_recommander"]["messages"][-1].content})
        if "calander" in c:
            result.append({"role": "calander","content": c["calander"]["messages"][-1].content})
    print(result)
    # user_history 의 길이가 10이 넘어가지 않도록 관리
    user_taste = llm.invoke(f""""
                            다음 대화 내용을 분석하여 사용자의 음식 취향을 추출합니다.  
                새로운 취향 정보가 확인되면 기존 취향 데이터({user_taste})를 업데이트합니다.  
                단, 사용자가 명확히 언급하지 않은 사항은 추가하지 않으며, 불확실한 정보는 기존 취향을 유지합니다.  
                사용자의 감정적 표현(예: "이건 별로야", "좋아하지 않음")과 선호(예: "좋아해", "자주 먹음")를 포함한 문맥을 고려하여 취향을 판단합니다.  
                사용자의 특정 상황(예: 다이어트, 채식, 특정 음식 알러지 등)도 반영하여 종합적으로 판단합니다.  

                ### 응답 예시 (실제 데이터와 다를 수 있음):
                **사용자의 취향:**  
                ✅ 선호하는 음식: 닭고기, 매운 음식, 초밥  
                ❌ 기피하는 음식: 해산물, 당도가 높은 음식  
                ⚠️ 상황적 요소: 다이어트 중, 저탄수화물 식단 유지  

                ### 대화 내역:
                {messages}  

                ### 기존 취향 데이터:
                {user_taste}  

                **업데이트된 사용자의 취향을 반환하세요.**  
                """).content
    rd.set(f"{user_id}_history", json.dumps(full_result + result))
    print("생성된 유저 취향: ", user_taste)
    # list 파일 redis
    rd.set(user_id, user_taste)
    return result
