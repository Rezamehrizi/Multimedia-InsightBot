import streamlit as st
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
# from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
load_dotenv()






# App title 
st.set_page_config(page_title=" Chatbot (with OpenAI)")

# Replicate Credentials (sidebar)
with st.sidebar:

    st.write("""
    **ConverseAI** is an innovative chatbox powered by LLMs. 
             
             
    **How does it work?** 🤔
             
    1. Enter your [OpenAI API Key](https://platform.openai.com/account/api-keys).
    """)
    if 'OPENAI_API_KEY' in st.secrets:
        st.success('API key already provided!', icon='✅')
    else:
        openai_api_key = st.text_input('Enter OpenAI API key:', type='password')
        os.environ['OPENAI_API_KEY'] = openai_api_key

    # Add a collapsible section for parameters
    with st.expander("Parameters"):
        temperature = st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        max_length = st.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    
    st.write("""
    2. Enter your file or URL.
    3. Start chatting with your file or website.
    """)

st.subheader("ConverseAI: Instant Conversations with Any Website", anchor=False)
st.write("""
ConverseAI is an innovative chatbox powered by LLMs. 
""")


# with st.expander("💡 Video Tutorial"):
#     with st.spinner("Loading video.."):
#         st.video("https://youtu.be/yzBr3L2BIto", format="video/mp4", start_time=0)

import streamlit as st

# Define available file types

def get_text_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    text = loader.load()
    return text

def get_text_from_pdf(uploaded_file):                    
    # pdf_reader = PdfReader(uploaded_file)
    # # store the pdf text in a text
    # text = ""
    # for page in pdf_reader.pages:
    #     text += page.extract_text() + "\n"
    loader = PyPDFLoader(uploaded_file)
    text = loader.load_and_split()
    return text

def get_text_from_csv(uploaded_file):
    loader = CSVLoader(uploaded_file)  
    text = loader.load()
    return text

def get_text_from_txt(uploaded_file):
    loader = TextLoader(uploaded_file)  
    text = loader.load()
    return text

# Create a multi-choice box
file_types = ["URL", "PDF", "TXT", "CSV"]
selected_file_type = st.selectbox("Select file type:", file_types, key="file_type")



# Display elements based on selection
if selected_file_type:
    if "URL" in selected_file_type:
        # Handle URL input
        url = st.text_input("Paste URL:")
        if url:
            st.write(f"You entered URL: {url}")
            document = get_text_from_url(url)
            
    else:
        # Handle file upload for non-URL types
        uploaded_file = st.file_uploader(f"Upload {selected_file_type} file:")
        if uploaded_file:
            st.write(f"You uploaded {uploaded_file.name}")
            # Process the uploaded file based on its type (e.g., read content, perform analysis)
            if "PDF" in selected_file_type:
                document = get_text_from_pdf(uploaded_file)                
                
            elif "TXT" in selected_file_type:
                document = get_text_from_txt(uploaded_file)
                
            else:
                document = get_text_from_csv(uploaded_file)

else:
    st.write("Please select at least one file type.")



def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, HuggingFaceEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5)
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']


# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How may I assist you today?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_url(url)    

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

