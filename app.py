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

# --- 1. INITIALIZATION & CONFIG ---
load_dotenv()

# LangSmith Tracing Setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Hybrid-RAG-Assistant"

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="wide")

# Path to your primary dataset
PRIORITY_PDF_PATH = os.path.join("dataset", "chatbot_dataset.pdf")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 2. DOCUMENT PROCESSING LOGIC ---
def build_vector_store(uploaded_files):
    all_docs = []
    
    # Load Priority Dataset automatically
    if os.path.exists(PRIORITY_PDF_PATH):
        try:
            loader = PyPDFLoader(PRIORITY_PDF_PATH)
            all_docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading priority dataset: {e}")
    
    # Load User Uploaded Files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # File Type Routing
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initial index if no retriever exists
if "retriever" not in st.session_state:
    st.session_state.retriever = build_vector_store([])

# --- 3. SIDEBAR UI ---
with st.sidebar:
    st.title("📁 Document Control")
    st.markdown(f"**Primary Source:** `{os.path.basename(PRIORITY_PDF_PATH)}`")
    
    user_uploads = st.file_uploader(
        "Add Extra Documents", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("Update Knowledge Base"):
        with st.spinner("Indexing documents..."):
            st.session_state.retriever = build_vector_store(user_uploads)
            st.success("Indexing Complete!")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- 4. CHAT INTERFACE & LOGIC ---
st.title("🤖 Hybrid AI Knowledge Assistant")
st.info("Searching priority dataset and uploaded files. Falling back to LLM if no answer is found.")

# Display Chat History
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Handle User Input
if prompt := st.chat_input("Ask me anything about the data..."):
    st.chat_message("user").markdown(prompt)
    final_response = ""

    with st.chat_message("assistant"):
        # PHASE 1: Retrieval from Vector DB
        if st.session_state.retriever:
            # 1. Define Prompt
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question using ONLY the context below. If the answer is not in the context, reply exactly with: 'NOT_FOUND_IN_DOCS'.\n\nContext: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            # 2. Retrieve Relevant Chunks (Corrected method: invoke)
            relevant_docs = st.session_state.retriever.invoke(prompt)
            context_text = format_docs(relevant_docs)

            # 3. Execute RAG Chain
            rag_chain = rag_prompt | st.session_state.llm | StrOutputParser()
            
            pdf_res = rag_chain.invoke({
                "context": context_text,
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })

            if "NOT_FOUND_IN_DOCS" not in pdf_res:
                final_response = pdf_res

        # PHASE 2: Fallback to General Knowledge
        if not final_response or "NOT_FOUND_IN_DOCS" in final_response:
            with st.status("Searching beyond documents (LangSmith Tracing active)..."):
                gen_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the user's question to the best of your general knowledge. Mention if you are not relying on the provided documents."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])
                gen_chain = gen_prompt | st.session_state.llm | StrOutputParser()
                final_response = f"**[General Knowledge]** {gen_chain.invoke({'input': prompt, 'chat_history': st.session_state.chat_history})}"

        # Final Display
        st.markdown(final_response)
    
    # Save History
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=final_response))
