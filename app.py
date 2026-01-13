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
from langchain_core.runnables import RunnablePassthrough

# --- 1. INITIALIZATION (Must be at the very top) ---
load_dotenv()
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="wide")

# Crucial: Initialize all session state keys before anything else
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

PRIORITY_PDF_PATH = os.path.join("dataset", "chatbot_dataset.pdf")

# --- 2. INDEXING LOGIC ---
def build_vector_store(uploaded_files):
    all_docs = []
    if os.path.exists(PRIORITY_PDF_PATH):
        try:
            all_docs.extend(PyPDFLoader(PRIORITY_PDF_PATH).load())
        except Exception as e:
            st.warning(f"Could not load priority dataset: {e}")
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(temp_path) if uploaded_file.name.endswith(".pdf") else \
                     Docx2txtLoader(temp_path) if uploaded_file.name.endswith(".docx") else \
                     TextLoader(temp_path)
            all_docs.extend(loader.load())
            os.remove(temp_path)

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs)
        vectorstore = FAISS.from_documents(splits, st.session_state.embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    return None

if "retriever" not in st.session_state:
    st.session_state.retriever = build_vector_store([])

# --- 3. UI SIDEBAR ---
with st.sidebar:
    st.title("📁 Document Control")
    user_uploads = st.file_uploader("Upload Extra Docs", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    if st.button("Update Knowledge"):
        st.session_state.retriever = build_vector_store(user_uploads)
        st.success("Indexed!")
    if st.button("Clear History"):
        st.session_state.chat_history = []
        st.rerun()

# --- 4. CHAT LOGIC ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("🤖 Hybrid AI Knowledge Assistant")

# Display previous messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask me something..."):
    st.chat_message("user").markdown(prompt)
    final_response = ""

    with st.chat_message("assistant"):
        # Step 1: PDF Search
        if st.session_state.retriever:
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer ONLY using context. If missing, say 'NOT_FOUND_IN_DOCS'.\n\nContext: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            # Use dictionary keys instead of lambda for st.session_state to avoid the AttributeError
            chain_input = {
                "context": st.session_state.retriever | format_docs,
                "input": RunnablePassthrough(),
                "chat_history": lambda x: st.session_state.chat_history # Passing history directly
            }
            
            rag_chain = rag_prompt | st.session_state.llm | StrOutputParser()
            
            # Invoke with specific keys
            pdf_res = rag_chain.invoke({
                "context": format_docs(st.session_state.retriever.get_relevant_documents(prompt)),
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            if "NOT_FOUND_IN_DOCS" not in pdf_res:
                final_response = pdf_res

        # Step 2: Fallback
        if not final_response or "NOT_FOUND_IN_DOCS" in final_response:
            with st.status("Consulting LLM Knowledge..."):
                gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer from general knowledge."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                gen_chain = gen_prompt | st.session_state.llm | StrOutputParser()
                final_response = f"**[General Knowledge]** {gen_chain.invoke({'input': prompt, 'chat_history': st.session_state.chat_history})}"
        
        st.markdown(final_response)
        
        # Save to history
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=final_response))
