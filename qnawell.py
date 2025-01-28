import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain


# Configuration and initialization
api_key = os.environ.get("MISTRAL_API_KEY")
print(api_key)

st.title("Interactive QnA with Drilling Reports")
# Load data
loader = TextLoader("/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/well_remark.txt")
docs = loader.load()
# Split text into chunks 
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
# Define the embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
# Create the vector store 
vector = FAISS.from_documents(documents, embeddings)
# Define a retriever interface
retriever = vector.as_retriever()
# Define LLM
model = ChatMistralAI(mistral_api_key=api_key)
# Define prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
# Create a retrieval chain to answer questions
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# User interface for asking questions
st.subheader("Ask a question about the well history")
question = st.text_area("Enter your question here", "")
if st.button("Ask"):
    response = retrieval_chain.invoke({"input": question})
    st.text("Response:")
    st.write(response.get("answer", "No response generated. Check the configuration and inputs."))

