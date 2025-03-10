from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.documents import Document

import requests
from datetime import datetime, timedelta

from app.config import settings

all_menus = []

retreiver = None

def fetch_menu_data(date):
    url = "https://mc.skhystec.com/V3/prc/selectMenuList.prc"
    headers = {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "x-requested-with": "XMLHttpRequest"
    }
    campus = "BD"
    cafeteria_seq = 21
    meal_type = "LN"
    ymd = date
    
    body = f"campus={campus}&cafeteriaSeq={cafeteria_seq}&mealType={meal_type}&ymd={ymd}"
    response = requests.post(url, headers=headers, data=body, verify=False)
    
    if response.status_code == 200:
        try:
            data = response.json()
            if "RESULT" in data and data["RESULT"] == 'N':
                data = None
            d = {"date": date, "body": data}
            all_menus.append(Document(page_content=str(d)))
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

def generate_embedding():
    global retriever
    embeddings = AzureOpenAIEmbeddings(
        model=settings.aoai_deploy_embed_3_large,
        openai_api_version="2024-02-01",
        api_key= settings.aoai_api_key,
        azure_endpoint=settings.aoai_endpoint
    )

    vectorstore = FAISS.from_documents(documents=all_menus, embedding=embeddings)
    retriever = vectorstore.as_retriever()

def generate_prompt(messages: str):
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Context: 
    {context}

    #Question:
    {question}

    #Answer:"""
    )

    llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_deployment=settings.aoai_deploy_gpt4o_mini,
        temperature=0.0,
        api_key= settings.aoai_api_key,  
        azure_endpoint=settings.aoai_endpoint
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = messages

    return chain.invoke(question)