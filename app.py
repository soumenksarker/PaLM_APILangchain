__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

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

def create_conversational_chain(vector_store):
    load_dotenv()
    # Create llm
    #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                        #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    # llm = Replicate(
    #     streaming = True,
    #     model = "replicate/llama-7b:ac808388e2e9d8ed35a5bf2eaa7d83f0ad53f9e3df31a42e4eb0a0c3249b3165", 
    #     callbacks=[StreamingStdOutCallbackHandler()],
    #     input = {"temperature": 0.01, "max_length" :500,"top_p":1})

    # chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
    #                                              retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    #                                              memory=memory)
    llm = GooglePalm(callbacks=[StreamingStdOutCallbackHandler()], temperature=0.1)  # OpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        # input_key="question",
        memory=memory,
        )
    return chain

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

        # Create embeddings
        embeddings = GooglePalmEmbeddings()

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()
