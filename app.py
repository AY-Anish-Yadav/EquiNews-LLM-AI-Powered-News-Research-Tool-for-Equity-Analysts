import os
import streamlit as st
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("Bot: Research Tool 📈")
st.sidebar.title("Upload your Doc in PDf Format")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file")
                                
process_url_clicked = st.sidebar.button("Process PDFs")
faiss_index_file = "faiss_store_openai.pkl"

main_placeholder = st.empty()
K=""
llm=OpenAI(openai_api_key=K)
embeddings=OpenAIEmbeddings(openai_api_key=K)

if process_url_clicked:
    # load data
    file_name = uploaded_file.name 
    pdf_loader = PyPDFLoader(file_name)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = pdf_loader.load()
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,)

    docs = text_splitter.split_documents(data)
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    with open(faiss_index_file,"wb") as f:
        pickle.dump(vectorstore_openai,f)
        main_placeholder.text("Embeddings are Stored...✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(faiss_index_file):
        with open(faiss_index_file,"rb") as f:
            vectorIndex=pickle.load(f)
            retriever = vectorIndex.as_retriever()
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=retriever)
            result=chain(query)
            st.header("Answer")
            st.write(result["answer"])


    
