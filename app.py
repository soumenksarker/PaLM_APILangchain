import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQA

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
import pinecone

import os
from dotenv import load_dotenv
import tempfile
import time

load_dotenv()


def initialize_session_state():
    if 'file' not in st.session_state:
        st.session_state['file'] = "no"
    if 'url' not in st.session_state:
        st.session_state['url'] = "no"

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything beyond your provided information!🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain):
    response = chain.run(query)
    #history.append((query, response))
    return response

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            _, c, _=st.columns([2,1,2])
            submit_button = c.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")
#@st.cache
def create_conversational_chain(docsearch):
    load_dotenv()
    prompt_template  = """
    Base on the following piece of context and searching from other sources update if needed to answer the question. Translate in Bangla language using Google Translate if the user asks for translation. Please provide a detailed response for each of the questions.
    
    {context}
    
    Question: {question}
    
    Answer in English, and Bangla based on the input query and user demands"""
    prompt = PromptTemplate(template = prompt_template , input_variables=["context", "question"])
    llm = GooglePalm(temperature=0.1) 
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    return qa
def load_embedding(text_chunks):
    load_dotenv()
    # # Create embeddings
    embeddings = GooglePalmEmbeddings()
    #query_result = embeddings.embed_query("Hello World")
    pinecone.init(api_key="0fa06a79-cf08-484b-b91c-236b77956235",  # find at app.pinecone.io
                  environment="gcp-starter")  # next to api key in console)
     # initialize pinecone
    index_name = "langchainpalm2pinecone" # put in the name of your pinecone index here
    docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
    chain = create_conversational_chain(docsearch)
    return chain
def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("Expert System/Information retrieval beyond your data:books:")

    def change_file_state():
        st.session_state.file="yes"
    def change_url_state():
        st.session_state.url="yes"
    
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload multiple files texts, docx or pdfs", accept_multiple_files=True, on_change=change_file_state)
    st.sidebar.title("Enter URLs")
    #urls = []
    url = st.sidebar.text_input(f"URL", on_change=change_url_state)
    #urls.append(url)
    st.sidebar.write("To extract info from multiple URL's, paste a new URL replacing the previous one, info from new URL will be accommodated to the system automatically!")
    #st.sidebar.button("Process")
    main_placeholder = st.empty()
    text = []
    text_chunks=[]
    if url and st.session_state.url=="yes":
        # load data
        loader = UnstructuredURLLoader(urls=[url])
        main_placeholder.text("System is loading ✅✅✅")
        text.extend(loader.load())
        # #time.sleep(2)
        # data = loader.load()
        # # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        #main_placeholder.text("Text Splitter...Started...✅✅✅")
        text_chunks = text_splitter.split_documents(text)
        text=[]
        st.session_state.url="no"
        # urls=[]
        #urls = []
    if uploaded_files and st.session_state.file=="yes":
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path) 
        main_placeholder.text("System is loading ✅✅✅")
        text_splitter =RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=700, chunk_overlap=150) 
        text_chunks = text_splitter.split_documents(text)
        text=[]
        st.session_state.file="no"
        
      
    # Create the chain object
    chain=load_embedding(text_chunks)
    main_placeholder.text("System is Running, Interact! ✅✅✅")
    display_chat_history(chain)

if __name__ == "__main__":
    main()

