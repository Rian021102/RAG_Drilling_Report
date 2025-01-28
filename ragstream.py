import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Define paths as constants
DATA_PATH = '/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/drilling_report_comp.csv'
TEXT_PATH = '/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/data/remark_test_02.txt'
DB_PATH = '/Users/rianrachmanto/miniforge3/project/RAG_Drill_Report/Chroma_DB'

st.title("Ask Anything About Your Drilling Reports!")
st.header("Extract information directly from your reports")

@st.cache(allow_output_mutation=True)
def setup_retrieval():
    # Load and split documents
    loader = TextLoader(TEXT_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
    documents = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="bge-m3")
    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=DB_PATH)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

def get_data(path):
    df = pd.read_csv(path)
    df = df[['Wellbore', 'Remark']]
    return df

def convert_to_text(wellbore_df):
    with open(TEXT_PATH, 'w') as f:  # Change mode to 'w' to overwrite each time
        df_string = wellbore_df.to_string(header=False, index=False)
        f.write(df_string)

def main():
    df = get_data(DATA_PATH)
    wellbore = st.sidebar.selectbox('Select Wellbore', df['Wellbore'].unique())
    wellbore_df = df[df['Wellbore'] == wellbore].copy()
    st.write(wellbore_df)

    if st.button('Load and Split Text'):  # Button to convert and load text
        convert_to_text(wellbore_df)

    retriever = setup_retrieval()  # Setup retriever only once

    # Set up the local model and RAG chain
    local_model = "llama3.2"
    llm = ChatOllama(model=local_model, num_predict=400, stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    prompt_template = """
    <|start_header_id|>user<|end_header_id|>
    You are a drilling engineer and an analyst. Your task is reading the drilling report,
    within the context of the drilling report, answer the question based on the context of the document.
    All you need to answer what you can find in the document. If you can't find the answer, you can say "I don't know".
    Question: {question}
    Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    question = st.text_input("Ask a question")
    if question:
        response = rag_chain.invoke({"question": question})
        answer = response.get('answer', 'No answer found in response') if isinstance(response, dict) else response
        st.text_area("Response", value=answer, height=200, disabled=True)

if __name__ == "__main__":
    main()
