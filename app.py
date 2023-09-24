import streamlit as st
from streamlit_chat import message
from langchain.chains import RetrievalQA

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


load_dotenv()


def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    
    
    #result = chain({"question": query, "chat_history": history})
    response = chain(query)
    history.append((query, response["result"]))
    return response["result"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(docsearch):
    load_dotenv()
    prompt_template  = """
    Use the following piece of context to answer the question. Please provide a detailed response for each of the question.
    
    {context}
    
    Question: {question}
    
    Answer in Italian"""
    prompt = PromptTemplate(template = prompt_template , input_variables=["context", "question"])
    llm = GooglePalm(temperature=0.1) 
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    return qa

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
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

        text_splitter =RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200) 
        text_chunks = text_splitter.split_documents(text)

        # # Create embeddings
        embeddings = GooglePalmEmbeddings()
        query_result = embeddings.embed_query("Hello World")
        st.write(len(query_result))
        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_API_ENV  # next to api key in console
        )
        index_name = "langchainpinecone" # put in the name of your pinecone index here

        docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
        # Create the chain object
        chain = create_conversational_chain(docsearch)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()

