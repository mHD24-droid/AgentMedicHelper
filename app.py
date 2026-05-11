import streamlit as st
from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Study Assistant", page_icon="🩺", layout="wide")

# --- CACHE MODELS (So they don't reload every click) ---
@st.cache_resource
def get_resources():
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
load_llm():
    return ChatGroq(
        temperature=0, 
        groq_api_key="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", 
        model_name="llama-3.1-70b-versatile"
    )    search = DuckDuckGoSearchRun()
    return embeddings, llm, search

embeddings_model, llm, search_tool = get_resources()

# --- LOAD VECTOR STORE ---
base_dir = Path(__file__).resolve().parent
db_dir = base_dir / "vectorstore" / "db_faiss"

if db_dir.exists():
    db = FAISS.load_local(str(db_dir), embeddings_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
else:
    st.error("Vector database not found. Please run 'python vector.py' first!")
    st.stop()

# --- UI HEADER ---
st.title("🩺 Advanced Medical Study Agent")
st.markdown("---")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt_input := st.chat_input("Ask a medical question (EN/FR/AR)..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.status("Searching PDFs and Web...", expanded=False) as status:
            # 1. Retrieval
            docs = retriever.invoke(prompt_input)
            pdf_context = "\n".join([d.page_content for d in docs])
            sources = list(set([Path(d.metadata.get('source', 'Unknown')).name for d in docs]))
            
            # 2. Web Search
            web_context = search_tool.run(prompt_input)
            status.update(label="Generating response...", state="running")

            # 3. Prompting
            sys_msg = (
                "You are a multilingual medical assistant. Structure your answer with these headers: "
                "# 📘 Concept Definition, # 🎓 University Lecture Details, # 🌐 Additional Medical Knowledge, "
                "# 🩺 Clinical Note, # ⚡ High-Yield Exam Points, # 🧠 Memory Aid, # 📌 Quick Summary. "
                "Respond in the same language as the user input."
            )
            
            full_prompt = f"System: {sys_msg}\n\nContext:\n{pdf_context}\n\nOnline:\n{web_context}\n\nQuestion: {prompt_input}"
            response = llm.invoke(full_prompt)
            status.update(label="Complete!", state="complete")

        st.markdown(response)
        if sources:
            st.caption(f"**Sources:** {', '.join(sources)}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})
