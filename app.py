import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
# --- 1. SETUP & LANGSMITH TRACING ---
load_dotenv()

# Tracing is handled automatically by LangChain via environment variables
st.set_page_config(page_title="Pro Knowledge Assistant", page_icon="🤖", layout="wide")

PRIORITY_PDF_PATH = r"dataset\chatbot_dataset.pdf"

# Initialize Models in Session State
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 2. INDEXING LOGIC ---
def build_vector_store(uploaded_files):
    all_docs = []
    
    # Load Primary Dataset
    if os.path.exists(PRIORITY_PDF_PATH):
        priority_loader = PyPDFLoader(PRIORITY_PDF_PATH)
        all_docs.extend(priority_loader.load())
    
    # Load User Files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            
            all_docs.extend(loader.load())
            os.remove(temp_path)

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs)
        vectorstore = FAISS.from_documents(splits, st.session_state.embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    return None

# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown(f"**Primary Data:** `{os.path.basename(PRIORITY_PDF_PATH)}`")
    
    user_uploads = st.file_uploader(
        "Add Extra Context", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("Rebuild Knowledge Base"):
        with st.spinner("Syncing datasets and starting trace..."):
            st.session_state.retriever = build_vector_store(user_uploads)
            st.success("Knowledge Base & LangSmith Sync Ready!")

    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

if "retriever" not in st.session_state:
    st.session_state.retriever = build_vector_store([])

# --- 4. CHAT INTERFACE ---
st.title("🤖 Pro Knowledge Assistant")
st.caption("Connected to LangSmith for real-time trace analysis.")

# Display Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Prompt Templates
pdf_prompt = ChatPromptTemplate.from_messages([
    ("system", "Use ONLY the context below. If missing, reply: 'NOT_FOUND_IN_DOCS'\n\nContext: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on general AI knowledge as the documents do not contain the answer."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

if prompt := st.chat_input("Ask me about the datasets..."):
    st.chat_message("user").markdown(prompt)
    response = ""

    with st.chat_message("assistant"):
        # Step A: Attempt Document Retrieval
        if st.session_state.retriever:
            doc_chain = create_stuff_documents_chain(st.session_state.llm, pdf_prompt)
            relevant_docs = st.session_state.retriever.invoke(prompt)
            pdf_res = doc_chain.invoke({
                "input": prompt, 
                "context": relevant_docs, 
                "chat_history": st.session_state.chat_history
            })
            if "NOT_FOUND_IN_DOCS" not in pdf_res:
                response = pdf_res

        # Step B: LLM Fallback
        if not response or "NOT_FOUND_IN_DOCS" in response:
            with st.status("Consulting general knowledge (Tracing in LangSmith)...", expanded=False):
                fallback_chain = gen_prompt | st.session_state.llm | StrOutputParser()
                response = f"**[General Knowledge]** {fallback_chain.invoke({'input': prompt, 'chat_history': st.session_state.chat_history})}"

        st.markdown(response)
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response))
