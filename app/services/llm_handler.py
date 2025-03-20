
import asyncio
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

# Global variables
chats_by_session_id = {}
vectorstores = {}

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    model=settings.aoai_deploy_embed_3_large,
    openai_api_version="2024-02-01",
    api_key=settings.aoai_api_key,
    azure_endpoint=settings.aoai_endpoint
)


# Initialize LLMs
## o3
o3_llm = AzureChatOpenAI(
    openai_api_version=settings.aoai_o3_mini_version,
    azure_deployment=settings.aoai_o3_mini_deployment_name,
    # temperature=0.0,
    api_key=settings.aoai_o3_mini_api_key,
    azure_endpoint=settings.aoai_o3_mini_endpoint
)

## 4o mini
llm = AzureChatOpenAI(
    openai_api_version=settings.aoai_deploy_gpt4o_version,
    azure_deployment=settings.aoai_deploy_gpt4o_mini,
    temperature=0.0,
    api_key=settings.aoai_api_key,
    azure_endpoint=settings.aoai_endpoint
)

# -------------------- Utility Functions --------------------
def load_vector_stores():
    """Load vector stores from disk if they exist"""
    for source in ["dining_code"]:
        path = f"{source}_faiss_index"
        if os.path.exists(path) and os.path.exists(f"{path}/index.pkl"):
            print(f"Loading existing vector store for {source} from disk...")
            try:
                vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                vectorstores[source] = vectorstore
                print(f"Successfully loaded {source} vector store")
            except Exception as e:
                print(f"Error loading {source} vector store: {e}")

def generate_embedding(source, menus):
    """Generate embeddings for a specific source and menus"""
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

def preprocess_query(query):
    """Remove negative statements and standardize query for vector search"""
    # Use LLM to extract just the positive intent
    processed_query = llm.invoke(
        f"다음 질문에서 부정적인 표현을 제거하고 검색어로 적합한 형태로 변환해주세요. 부정적인 표현을 긍정적인 표현으로 변경 금지합니다. 해당 없으면 원문 그대로 반환: '{query}'"
    ).content
    return processed_query

# -------------------- Tool Definitions --------------------
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
def get_dining_code_menus(message: Annotated[str, "질의 내용 중 다이닝코드(dining_code) VectorDB 에서 유사도 검색이 필요할 것들에 대한 질의. 부정적인 질의는 검색에 방해가 되므로 이 단계에서는 제외한다."]):
    """메뉴에 대해서 검색하고 실제 있는 값인지 확인한다."""
    processed_message = preprocess_query(message)
    print(f"Original: {message}\nProcessed: {processed_message}")
    
    if "dining_code" not in vectorstores:
        return "No retriever available. Please try again later."
    
    r = vectorstores["dining_code"].similarity_search(processed_message, k = 10)
    
    return r

@tool
def get_exact_dining_code_data(v_rid: Annotated[str, "질의 내용 중 다이닝코드(dining_code) 사이트에서 정확한 값이 필요한 경우 restaurant_id 값을 활용해 제공한다."]):
    """매장에 대해서 검색하고 상세 정보를 확인한다."""
    # print(v_rid)
    return fetch_exact_dining_code_data(v_rid)

# -------------------- Agent Definitions --------------------
members = ["hyteria_menu_retriever", "calander", "dining_code_menu_retriever", "menu_recommander"]
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
    "~~집 은 dining_code 를 참조하세요."
    "구내식당 메뉴를 찾기 전 calenar 를 참조하세요."
    "기본 응답 언어는 한국어입니다."
    "If date related question is asked, You Should use calander worker before other workers. "
    f"If requests doesn't include any specific menu or restaurants, You Should use all of menu retriever worker."
    "You Should use calander worker before hyteria_menu_retriever worker. "
    "You Should use hyteria_menu_retriever worker if user'll ask about hyteria menu. "
    "You Should use dining_code_menu_retriever worker if user'll ask about dining_code menu or eat outside. "
    "You Should use menu_recommander worker if user'll ask about recommand menu. And You should use menu_recommander worker after hyteria_menu_retriever and dining_code_menu_retriever. "
    "menu_recommander **MUST** be used after hyteria_menu_retriever or dining_code_menu_retriever."
    "If no more workers are needed, respond with FINISH."
    "When finished, respond with FINISH."
)

class State(MessagesState):
    next: str

# -------------------- Agent Nodes --------------------
def supervisor_node(state: State) -> Command[Literal[*members[1:]]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})

# Create the specialized agents
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
    , prompt="You are restaurant retriever. You can check all menu on following date with Restaurant VectorDB. Do Not Math. Do not recommend restaurant. 답변에 추가적인 의견을 제공하지 말고 레스토랑 정보들만 제공해주세요. 사용자의 요청의 적합한 레스토랑만 가져오세요. 모든 레스토랑을 원하는 경우 모든 레스토랑을 가져와주세요. 정보를 변형하지말고 정확하게 전달해주세요. 다른 에이전트에서 전달 받은 값과 상관없이 dining_code vector DB 의 값을 조회해주세요. Don't recommand. 식당 추천 금지. 식당 검색 후 menu_recommander 에게 전달해주세요."
    , response_format=DiningCodeRestaurantOutputList
)

menu_recommander_agent = create_react_agent(
    o3_llm, tools=[get_exact_dining_code_data], 
    prompt="""You are menu recommander. 참석자를 생각해보고 참석자가 선호할만한 음식들을 생각해보자. 
    그리고 답변에 추천의 이유를 상세히 작성해서 논리적으로 설명해주세요.
    다른 agent에 의해 전달받은 값이 없으면 답을 줄 수 없습니다. 사용자가 원하는 메뉴를 주어진 메뉴들 중에 골라주세요.
    만약 면요리에 대해서 물어본다면 국수, 라면, 파스타 등에 대한 정보를 찾아주세요. 또한, 답변은 상세하게 진행해주세요.
    답변에 대한 추가적인 의견을 제공하지 말고 메뉴 정보들만 제공해주세요.
    리스트에 restaurant_id 가 있으면 그에 대한 정보를 가져와주세요. 정보를 변형하지말고 정확하게 전달해주세요.
    출처가 다이닝코드인 값은 반드시 툴을 활용해 상세 정보를 조회하세요. 질의에 restaurant_id 값이 존재한다면 상세 조회 툴을 사용하세요.
    알 수 없는 내용에 대해서는 모른다고 대답하세요.
    답변은 마크다운문법으로 예쁘게 만들어주세요.
    식당 메뉴가 있다면 제공하고 이미지가 있다면 보여주세요.
    충분히 고민해보고 대답하세요.
    추천하는 이유들을 상세히 설명해주고 또한, 사용자의 질문에 적절한 추천인지도 생각해보세요.
    """
)

def calander_node(state: State) -> Command[Literal["hyteria_menu_retriever"]]:
    
    result = calander_agent.invoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["structured_response"].date, name="calander")
            ]
        },
        goto="hyteria_menu_retriever"
    )

def hyteria_menu_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    if "messages" not in state or not state["messages"]:
        return Command(goto="calander")
    date = state["messages"][-1].content
    # validate date with regex if it is not in the correct format(YYYY-MM-DD) go to calander node
    if not re.match(r"\d{4}-\d{2}-\d{2}", date):
        return Command(goto="calander")
    
    content = get_hyteria_menus(date)

    # Process the content to extract date information
    # You can use an LLM call here to parse the content into structured data if needed
    try:
        # Using llm to parse the calendar content into a structured format
        processed_content = llm.invoke(
            f"""
            아래 {content}에 있는 리스트를 마크다운 문법으로 보기 좋게 작성해 주세요.
            목록 사이에 top bottom margin 이나 padding 주세요.
            줄 바꿈은(\n) \n\n 으로 해주세요.
            주의사항:

            마크다운 문법을 사용할 때 markdown 코드 블록은 사용하지 말고 일반 텍스트 형태로 작성합니다.
            메뉴 이미지의 파일 이름 예시:
            파일명: 20250313_BD_2_4_LN_1_20250313111249_0.jpg
            이 경우, 메뉴 이미지의 베이스 URL은 https://mc.skhystec.com/nsf/menuImage/20250313/BD/2/4/가 됩니다.
            즉, 파일명에서 첫 네 개의 밑줄(_)로 구분된 토큰을 추출하여, URL의 경로로 사용합니다.
            이 규칙을 반영해서 리스트를 마크다운 형식으로 예쁘게 작성해 주세요.
            """
        )
        
        formatted_response = processed_content.content
    except Exception as e:
        print(f"Error parsing data response: {e}")
        formatted_response = content  # Fallback to original content
    
    
    # print(formatted_response)
    return Command(
        update={
            "messages": [
                HumanMessage(content=str(formatted_response), name="hyteria_menu_retriever"),
            ]
        },
        goto="supervisor"
    )

def dining_code_menu_retriever_node(state: State) -> Command[Literal["menu_recommander"]]:
    result = dining_code_menus_retriever_agent.invoke(state)

    # Extract the response content
    content = result["structured_response"].restaurants

    # Process the content to extract date information
    # You can use an LLM call here to parse the content into structured data if needed
    try:
        # Using llm to parse the calendar content into a structured format
        processed_content = llm.invoke(
            f"다음 리스트를 마크다운문법으로 예쁘게 만들어주세요. 코드 블럭은 사용하지 말아 주세요: {content}"
        )
        
        formatted_response = processed_content.content
    except Exception as e:
        print(f"Error parsing data response: {e}")
        formatted_response = content  # Fallback to original content

    return Command(
        update={
            "messages": [
                HumanMessage(content=str(formatted_response), name="dining_code_menu_retriever"),
            ]
        },
        goto="menu_recommander"
    )

def menu_recommander_node(state: State) -> Command[Literal[END]]:
    result = menu_recommander_agent.invoke(state)

    content = result["messages"][-1].content

    formatted_response = content

    return Command(
        update={
            "messages": [
                HumanMessage(content=formatted_response, name="menu_recommander")
            ]
        },
        goto="supervisor"
    )

# -------------------- Initialize Graph --------------------
builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("calander", calander_node)
builder.add_node("hyteria_menu_retriever", hyteria_menu_retriever_node)
builder.add_node("dining_code_menu_retriever", dining_code_menu_retriever_node)
builder.add_node("menu_recommander", menu_recommander_node)
graph = builder.compile()

# -------------------- Main Function --------------------
async def generate_prompt(messages: str, user_id: str):
    s_prompt = SystemMessagePromptTemplate.from_template(
        f"""# 기본 원칙
            항상 오늘 날짜(예: 2025-03-20)를 기준으로 메뉴 정보를 확인하고, 메뉴 추천 시 날짜 일치를 최우선으로 고려합니다.
            사용자가 별도로 구내식당 메뉴를 요청하지 않는 한, 기본적으로 밖에서 먹고 싶다는 의미로 해석하여 메뉴를 추천합니다.
            # 메뉴 추천 조건
            오늘의 메뉴 데이터를 참조하여, 해당 날짜에 제공 가능한 메뉴 중에서 최적의 선택지를 추천합니다.
            사용자가 "구내식당"이라는 단어를 포함한 문의를 할 경우, 구내식당 메뉴 데이터를 기반으로 추천합니다.
            메뉴 추천 시, 메뉴의 맛, 영양, 계절성 등 다양한 요소를 고려하여 간략한 설명을 덧붙입니다.
            # 입력 처리
            사용자가 "오늘" 또는 특정 날짜(예: 2025-03-20)의 메뉴를 요청하면, 반드시 해당 날짜의 메뉴를 확인하고 추천합니다.
            메뉴 추천이 아닌 단순 날짜 문의일 경우, 오늘의 날짜(예: 2025-03-20)를 정확하게 답변합니다.
            # 출력 형식
            최종 결과는 마크다운 문법을 활용해 코드블럭 없이 텍스트 형태로 출력합니다.
            출력 내용은 오늘 날짜, 추천 메뉴, 그리고 메뉴에 대한 간략한 설명 등을 포함하여 제공됩니다.
            한국어를 통해 자연스럽게 응답하고, 사용자의 요청에 최대한 맞추어 답변합니다.
        """)
    u_prompt = HumanMessagePromptTemplate.from_template(messages)

    result = []

    # messages = [{chat_prompt.format()}]

    for c in graph.stream({"messages": [s_prompt.format(), u_prompt.format()]}, stream_mode="values"):
        # chunk에서 messages의 마지막 항목을 추출
        print("chunk: ", c["messages"][-1].name, c["messages"][-1].content)
        chat = {"role": c["messages"][-1].name, "content": c["messages"][-1].content}
        result.append({"role": c["messages"][-1].name, "content": c["messages"][-1].content})

        yield f"data: {str(json.dumps(chat))}\n\n"
        await asyncio.sleep(0.01)
    yield "event: end\ndata: [DONE]\n\n"
    # return result

load_vector_stores()