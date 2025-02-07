import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os


# To run locally:
# 1. Install the required packages:
#     pip install -r requirements.txt
# 2. Run the Streamlit app:
#     streamlit run pdf_chatbot.py


# Set OpenAI API key from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon=":alien:")
st.title("Ask Your Friendly Alien - PDF chat 👽")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o-mini')
    
    template = """You are a helpful AI assistant that helps users understand their PDF documents.
    You are a alien and add some funny alien emojis and witty lines to your messages.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know in a funny way, answer differently everytime, don't try to make up an answer.
    
    
    context: {context}

    chat history: {chat_history}
    
    Question: {question}
    Helpful Answer:"""

    # for some reason, {context} variable has to be injected for the prompt to work, {chat_history} variable works directly from the session state

    prompt = PromptTemplate(input_variables=['context', 'chat_history','question'], template=template)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return conversation_chain

def process_docs(pdf_docs):
    try:
        # Get PDF text
        raw_text = get_pdf_text(pdf_docs)
        
        # Get text chunks
        text_chunks = get_text_chunks(raw_text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector store using FAISS
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        
        # Create conversation chain
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.session_state.processComplete = True
        
        return True
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        return False
    
def display_chat_history():
    # Display chat history
    for role, avatar, message in st.session_state.chat_history:
        with st.chat_message(role, avatar=avatar):
            st.write(message)

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here",
        type="pdf",
        accept_multiple_files=True
    )
    
    if not st.session_state.processComplete and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)
            if success:
                st.success("Processing complete!")

# Main chat interface
if st.session_state.processComplete:
    user_question = st.chat_input("Ask any question about your Document(s)...")
    
    if user_question:
        user_role = "User"
        user_avatar = "🧑‍💻"
        st.session_state.chat_history.append((user_role, user_avatar, user_question)) # add question without waiting for answers
        # events like text input will rerun the app, hence the session state to preserve chat history, we are displaying 
        # the chat history and the last question asked by the user
        display_chat_history() 

        try:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                alien_role = "Alien"
                alien_avatar = "👽"
                st.session_state.chat_history.append((alien_role, alien_avatar, response["answer"]))
                # we are write the message directly instead of calling display_chat_history() to 
                # avoid displaying the last question twice
                with st.chat_message(alien_role, avatar=alien_avatar):
                    st.write(response["answer"])
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")

# Display initial instructions
else:
    st.write("👈 Upload your PDF(s) in the sidebar to get started!")
# Display initial instructions
else:
    st.write("👈 Upload your PDF(s) in the sidebar to get started!")
