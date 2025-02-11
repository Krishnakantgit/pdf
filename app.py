import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import time
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("API Key not found! Ensure .env file exists and contains the key.")
    st.stop()  # Stop execution if API key is missing

st.title("LLama based Document Assistant")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        # Load embeddings model
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load PDF documents from local file path
        st.session_state.loader = PyPDFLoader(file_path)
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.write("Vector Store DB is ready!")

# File uploader for user to upload their own PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_file_path = f"/tmp/{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process Document"):
        vector_embedding(temp_file_path)
        st.write("Document processed successfully!")

# Input box for the user question
prompt1 = st.text_input("Enter Your Question from the Document")

if prompt1:
    if "vectors" not in st.session_state:
        st.error("Please upload and process a PDF first!")
    else:
        # Create retrieval and document chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Process the user's query
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start} seconds")
        st.write(response['answer'])

        # Display relevant document chunks
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
