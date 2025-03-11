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
from app.shared_state import all_menus

retriever = None

def generate_embedding():
    global retriever
    print("Generating embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        model=settings.aoai_deploy_embed_3_large,
        openai_api_version="2024-02-01",
        api_key=settings.aoai_api_key,
        azure_endpoint=settings.aoai_endpoint
    )
    print("Embeddings created:", embeddings)
    vectorstore = FAISS.from_documents(documents=all_menus, embedding=embeddings)
    print("Vectorstore created:", vectorstore)
    retriever = vectorstore.as_retriever()
    print("Retriever created:", retriever)

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
def get_menus(message: Annotated[str, "질의 내용 중 VectorDB 에서 유사도 검색이 필요할 것들에 대한 질의"]):
    """메뉴에 대해서 검색하고 실제 있는 값인지 확인한다."""
    return retriever.invoke(message)

# Create Agent Supervisor
members = ["calander", "menu_retriever", "menu_recommander"]

# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]

llm = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_deployment=settings.aoai_deploy_gpt4o_mini,
    temperature=0.0,
    api_key=settings.aoai_api_key,
    azure_endpoint=settings.aoai_endpoint
)

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. You Should use calander worker if user'll ask about date relate. When finished,"
    " respond with FINISH."
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
    , prompt="You are a calander master. You can retrieve today date, previous date and next date. Do not recommend menu."
)

menu_retriever_agent = create_react_agent(
    llm, tools=[get_menus], prompt="You are menu retriever. You can check all menu on following date with Menu VectorDB. Do Not Math. 정확히 주어진 날짜에 대한 음식만 찾아. Do not recommend menu."
)

menu_recommander_agent = create_react_agent(
    llm, tools=[], prompt="You are menu recommander. 사용자가 원하는 메뉴를 주어진 메뉴들 중에 골라주세요. 만약 면요리에 대해서 물어본다면 국수, 라면, 파스타 등에 대한 정보를 찾아주세요."
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

def menu_retriever_node(state: State) -> Command[Literal["supervisor"]]:
    result = menu_retriever_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="menu_retriever")
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
builder.add_node("menu_retriever", menu_retriever_node)
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
        Answer in Korean.

        #Question:
        {messages}

        #Answer:"""
    )

    result = []
    for c in graph.stream({"messages": [("user", prompt.format())]}):
        # chunk에서 messages의 마지막 항목을 추출 (응답이 여기에 있다고 가정)
        print(c)
        if "menu_retriever" in c:
            result.append(c["menu_retriever"]["messages"][-1].content)
        if "menu_recommander" in c:
            result.append(c["menu_recommander"]["messages"][-1].content)
        if "calander" in c:
            result.append(c["calander"]["messages"][-1].content)
    return "".join(result)
