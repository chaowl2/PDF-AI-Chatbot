# Importing necessary libraries and functions
import streamlit as st
import configparser
import tempfile
import os
import fitz
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS

# Azure OpenAI API Details
config = configparser.RawConfigParser() 
config.read('you_file_path_to_pdfs')
cred = dict(config.items('Azure Deployment'))
apiKey = cred['apikey']
base_url = cred['base_url']
deployment_name = cred['deployment_name']
deployment_name2 = cred['deployment_name2']
deployment_name3 = cred['deployment_name3']
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_VERSION'] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = base_url
os.environ['OPENAI_API_KEY'] = apiKey

# Extracting texts from pdf
def get_pdf_text(uploaded_file):
    # create temp file path
    temp_file_path = os.getcwd()

    while uploaded_file is None:
        x = 1
# join temp file path and file name
    if uploaded_file is not None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())

# creating directory for tempfile and opening
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    docs = fitz.open(temp_file_path)

# iterates through docs, adds a marker for the start of the page and index and appened context to string
    text = ""
    for i, page in enumerate(docs):
        text += "<==({})==>\n".format(i+1)
        text += page.get_text()

# writing text to contract.txt
    with open("data/contract.txt", "w") as text_file:
        text_file.write(text)

# loading text to doc variable for chunking
    loader = TextLoader('data/contract.txt')
    doc = loader.load()
    return doc

# chunking texts using langchain tool
def get_text_chunks(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
    chunks = text_splitter.split_documents(doc)
    return chunks

# creating vector store using OpenAI embeddings and FAISS
def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model=deployment_name3,chunk_size=1)
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore

# creating coonversation chain and adding memory
def create_conversation_chain(vectorstore):
    llm = AzureChatOpenAI(deployment_name = deployment_name2,model_name = deployment_name2, temperature = 0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# handling user input using session_state function in streamlit
# see docs for more info about session_state in streamlit docs
def handle_input(user_question):
    response = st.session_state.chat({'question': user_question})
    st.session_state.chat_history = response['chat_history']
        
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**{message.content}**")
        else:
            st.write(message.content)

def main():
    # setting name on the tab
    st.set_page_config(page_title='🦜🔗 Contract Question and Answering using Langchain')

    if "chat" not in st.session_state:
        st.session_state.chat = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # setting page header
    st.header("🦜🔗 Chat with PDF")
    user_question = st.text_input('Ask a question about the contract:')
    if user_question:
        handle_input(user_question)

    with st.sidebar:
        st.subheader('Contracts')
        uploaded_file = st.file_uploader('Upload a Contract', type= 'pdf')
        
        if st.button("Proccess"):
            with st.spinner('Processing'):
                # getting raw text from pdf
                raw_text = get_pdf_text(uploaded_file)
                # getting chunks from raw text
                text_chunks = get_text_chunks(raw_text)
                # creating vectorstore 
                vectorstore = create_vectorstore(text_chunks)
                # creating conversation chain
                st.session_state.chat = create_conversation_chain(vectorstore)
                # after clicking procces button if it runs smoothly, then it will show "Ready".
                st.write('Ready') 

if __name__ == '__main__':
    main()
