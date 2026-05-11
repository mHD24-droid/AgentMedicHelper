import streamlit as st
from pathlib import Path
# REMOVE: from langchain_community.llms import Ollama
from langchain_groq import ChatGroq  # ADD THIS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Study Assistant", page_icon="🩺", layout="wide")

# --- CACHE MODELS ---
@st.cache_resource
def get_resources():
    # Keep multilingual embeddings for retrieval
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Replace Ollama with Groq
    # PRO TIP: Use st.secrets so your API Key isn't public on GitHub
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.1-70b-versatile"
    )
    
    search = DuckDuckGoSearchRun()
    return embeddings, llm, search

embeddings_model, llm, search_tool = get_resources()
