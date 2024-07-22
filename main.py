import streamlit as st
from readers import get_uploaded_file_paths, pdf_reader, prepare_vectorstore
from chains import  create_document_chain,chain

st.title("Teacher")
st.header("Upload PDF Files")

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

if uploaded_files:
    file_paths = get_uploaded_file_paths(uploaded_files)
    st.success(f"Uploaded {len(file_paths)} files successfully.")
    
    docs = pdf_reader(file_paths)
    prepare_vectorstore(docs)
    st.success("Documents processed and vector store created.")



st.header("Interface")
input_query = st.text_input("Enter your query")

if st.button("Submit"):
    if input_query:
        with st.spinner("Processing..."):
            chatbot = chain()
            result = chatbot.invoke(input_query)
            st.text(result)
    else:
        st.warning("Please enter a query.")
