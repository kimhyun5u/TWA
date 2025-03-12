from datetime import datetime, timedelta
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.config import settings

from app.shared_state import hyteria_menus
from app.shared_state import dining_code_menus

import os
from langchain_core.documents import Document

# Initialize the retriever
retrievers = {}

embeddings = AzureOpenAIEmbeddings(
    model=settings.aoai_deploy_embed_3_large,
    openai_api_version="2024-02-01",
    api_key=settings.aoai_api_key,
    azure_endpoint=settings.aoai_endpoint
)

# o3
# llm = AzureChatOpenAI(
#     openai_api_version=settings.aoai_o3_mini_version,
#     azure_deployment=settings.aoai_o3_mini_deployment_name,
#     # temperature=0.0,
#     api_key=settings.aoai_o3_mini_api_key,
#     azure_endpoint=settings.aoai_o3_mini_endpoint
# )

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
        retrievers[source] = vectorstore.as_retriever()
    else:
        # Vector store doesn't exist yet
        print("No existing vector store found. Will create when needed.")

def generate_embedding(source, menus):
    menus = [Document(page_content=str(menu)) for menu in menus]

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
    retrievers[source] = vectorstore.as_retriever()
    # print("Retriever created:", retriever)

def generate_embeddings():
    for source, menus in [("hyteria", hyteria_menus), ("dining_code", dining_code_menus)]:
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
def get_hyteria_menus(message: Annotated[str, "질의 내용 중 하이테리아(구내식당, hyteria) VectorDB 에서 유사도 검색이 필요할 것들에 대한 질의"]):
    """메뉴에 대해서 검색하고 실제 있는 값인지 확인한다."""
    if "hyteria" not in retrievers:
        return "No retriever available. Please try again later."
    return retrievers["hyteria"].invoke(message)

@tool
def get_dining_code_menus(message: Annotated[str, "질의 내용 중 다이닝코드(dining_code) VectorDB 에서 유사도 검색이 필요할 것들에 대한 질의"]):
    """메뉴에 대해서 검색하고 실제 있는 값인지 확인한다."""
    if "dining_code" not in retrievers:
        return "No retriever available. Please try again later."
    return retrievers["dining_code"].invoke(message)


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
    " task and respond with their results and status. You Should use calander worker if user'll ask about date relate. "
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
    llm, tools=[get_next_date, get_previous_date, get_today_date]
    , prompt="You are a calander master. You can retrieve today date, previous date and next date. Do not recommend menu. 답변에 대한 추가적인 의견을 제공하지 말고 날짜만 답변에 포함해주세요."
)

hyteria_menu_retriever_agent = create_react_agent(
    llm, tools=[get_hyteria_menus], prompt="You are menu retriever. You can check all menu on following date with Menu VectorDB. Do Not Math. 정확히 주어진 날짜에 대한 음식만 찾아. Do not recommend menu. 답변에 추가적인 의견을 제공하지 말고 메뉴 정보들만 제공해주세요. 정보를 변형하지말고 정확하게 전달해주세요."
)

dining_code_menus_retriever_agent = create_react_agent(
    llm, tools=[get_dining_code_menus], prompt="You are restaurant retriever. You can check all menu on following date with Restaurant VectorDB. Do Not Math. Do not recommend restaurant. 답변에 추가적인 의견을 제공하지 말고 레스토랑 정보들만 제공해주세요. 사용자의 요청의 적합한 레스토랑만 가져오세요. 모든 레스토랑을 원하는 경우 모든 레스토랑을 가져와주세요. 정보를 변형하지말고 정확하게 전달해주세요. 다른 에이전트에서 전달 받은 값과 상관없이 dining_code vector DB 의 값을 조회해주세요."
)

menu_recommander_agent = create_react_agent(
    llm, tools=[], prompt="You are menu recommander. 다른 agent에 의해 전달받은 값이 없으면 답을 줄 수 없습니다. 사용자가 원하는 메뉴를 주어진 메뉴들 중에 골라주세요. 만약 면요리에 대해서 물어본다면 국수, 라면, 파스타 등에 대한 정보를 찾아주세요. 또한, 답변은 상세하게 진행해주세요. 하이테리아는 hyteria 입니다. 다이닝코드는 dining_code 입니다. 구내식당(hyteria)와 외부식당(dining_code)의 메뉴를 모두 고려해주세요. 답변에 대한 추가적인 의견을 제공하지 말고 메뉴 정보들만 제공해주세요."
)

def calander_node(state: State) -> Command[Literal["supervisor"]]:
    result = calander_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="calander")
            ]
        },
        goto="supervisor"
    )

def hyteria_menu_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    result = hyteria_menu_retriever_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="hyteria_menu_retriever")
            ]
        },
        goto="supervisor"
    )

def dining_code_menu_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    result = dining_code_menus_retriever_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="dining_code_menu_retriever")
            ]
        },
        goto="supervisor"
    )

def menu_recommander_node(state: State) -> Command[Literal["supervisor"]]:
    result = menu_recommander_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="menu_recommander")
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

def generate_prompt(messages: str):
    prompt = PromptTemplate.from_template(
        f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Today is {datetime.now().strftime('%Y-%m-%d')}.
        Think about menus of {datetime.now().strftime('%Y-%m-%d')} and answer the following question.
        And Should think about the menu's date.
        What's date of today? And Think about the date in {messages}.
        만약 메뉴 추천을 원한다면 {datetime.now().strftime('%Y-%m-%d')}의 메뉴들을 찾아보고, 그 중에서 적합한 메뉴를 추천해주세요.
        단순 메뉴를 나열하는 것이 아닌 추천을 요청받으면 반드시 추천해주세요.
        Answer in Korean.

        #Question:
        {messages}

        #Answer:"""
    )

    result = []
    for c in graph.stream({"messages": [("user", prompt.format())]}):
        # chunk에서 messages의 마지막 항목을 추출 (응답이 여기에 있다고 가정)
        print(c)
        if "hyteria_menu_retriever" in c:
            result.append({"role": "hyteria_menu_retriever","content": c["hyteria_menu_retriever"]["messages"][-1].content})
        if "dining_code_menu_retriever" in c:
            result.append({"role": "dining_code_menu_retriever","content": c["dining_code_menu_retriever"]["messages"][-1].content})
        if "menu_recommander" in c:
            result.append({"role": "menu_recommander","content": c["menu_recommander"]["messages"][-1].content})
        if "calander" in c:
            result.append({"role": "calander","content": c["calander"]["messages"][-1].content})
    return result
