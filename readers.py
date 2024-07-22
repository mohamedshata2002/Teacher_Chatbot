import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google_api import api_key

def get_uploaded_file_paths(uploaded_files):
    file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_paths.append(tmp_file.name)
    return file_paths

def pdf_reader(file_paths):
    docs = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        doc = loader.load()
        docs.extend(doc)
    return docs

def prepare_vectorstore(docs):
    directory = "datastore"
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    split_docs = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=directory)
