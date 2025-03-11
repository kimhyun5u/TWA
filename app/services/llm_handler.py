from datetime import datetime, timedelta
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from app.config import settings
from app.shared_state import all_menus

retriever = None

llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_deployment=settings.aoai_deploy_gpt4o_mini,
        temperature=0.0,
        api_key=settings.aoai_api_key,
        azure_endpoint=settings.aoai_endpoint
    )
    
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

def generate_prompt(messages: str):
    prompt = PromptTemplate.from_template(
        f"""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Today is {datetime.now().strftime('%Y-%m-%d')}.
        Think about menus of {datetime.now().strftime('%Y-%m-%d')} and answer the following question.
        And Should think about the menu's date.
        Answer in Korean.

        #Context: 
        {{context}}

        #Question:
        {{question}}

        #Answer:"""

        #Answer:"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(messages)
