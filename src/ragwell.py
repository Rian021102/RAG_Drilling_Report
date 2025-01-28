from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
import time
import pandas as pd
import os

api_key = os.getenv("MISTRAL_API_KEY")

def loader_text(path):
    loader = TextLoader(path)
    docs = loader.load()
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    documents = text_splitter.split_documents(docs)
    return documents

def create_db(documents, embeddings):
    db = Chroma.from_documents(documents=documents, embedding=embeddings,persist_directory='/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/Chroma_DB_Fresh')
    return db

def create_retriever(db):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

def create_llm(local_model):
    llm = ChatOllama(model=local_model, num_predict=400,
                 stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    return llm

def create_rag_chain(retriever, llm):
    prompt_template = """
    <|start_header_id|>user<|end_header_id|>
    You are Workover engineer working at an offshore oil and gas company.You are tasked to read all workover report.
    Your job is to analyze the report and when asked, answer the question within the context of the report.
    For example when asked "What is the problem with the well?", You will answer within the context of the report.
    If you can not find any answer the within the contextual of the report just answer, "I don't know".
    Question: {question}
    Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    start_time = time.time()
    path="/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/well_remark.txt"
    docs = loader_text(path)
    documents = split_text(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = create_db(documents, embeddings)
    retriever = create_retriever(db)
    local_model = ChatMistralAI(mistral_api_key=api_key)
    llm = create_llm(local_model)
    rag_chain = create_rag_chain(retriever, llm)
    question = "Is there any obstruction discovered in the well? Specify the activity when the obstruction was discovered."
    print(question)
    print(rag_chain.invoke(question))
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()


