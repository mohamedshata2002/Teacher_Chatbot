from langchain.chains import LLMChain, ConversationalRetrievalChain,RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google_api import api_key
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import  ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM and Embeddings
llm = GoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")

def create_document_chain():
    vectorstore = Chroma(persist_directory="datastore", embedding_function=embeddings)
    llm_extractor = LLMChainExtractor.from_llm(llm)
    # retriever = ContextualCompressionRetriever(
    #     base_compressor=llm_extractor, 
    #     base_retriever=vectorstore.as_retriever()
    # )
    chain =RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=vectorstore.as_retriever(), 
        
    )
    return chain
def chain():
    vectorstore = Chroma(persist_directory="datastore", embedding_function=embeddings)
    reteriver = vectorstore.as_retriever()
    anwer = ({"question": RunnablePassthrough(),"context":reteriver}|
         ChatPromptTemplate.from_template("Answer the the question from the context  The context: {context} The question: {question}")|
         llm|StrOutputParser())
    return anwer
    

