from pathlib import Path
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize Search Tool
search = DuckDuckGoSearchRun()

def load_llm():
    return Ollama(model="llama3.2")

def main():
    base_dir = Path(__file__).resolve().parent
    db_dir = base_dir / "vectorstore" / "db_faiss"
    
    # Use the same multilingual embeddings as vector.py
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    db = FAISS.load_local(str(db_dir), embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    system_prompt = (
        "You are a multilingual medical assistant. Use the context provided to answer. "
        "If the answer is not in the context, use your internal knowledge but mention it. "
        "Always respond in the same language as the user's question (French, Arabic, or English).\n\n"
        "Concept Definition"
        "Brief, clear explanation."
        "Highest priority summaries from your PDFs."
        "Verified facts from online/general sources (clearly labeled)."
        "Real-world application or diagnosis/treatment."
        "Key Fact: Crucial for exams."
        "Easy trick for recall."
        "3–5 bullet points for rapid revision."
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def rag_chain(user_input):
        # 1. Search Local PDF Context
        docs = retriever.invoke(user_input)
        sources = list(set([doc.metadata.get('source', 'Unknown PDF') for doc in docs]))
        context = "\n".join([doc.page_content for doc in docs])
        
        # 2. Decide if we need Web Search
        # If local context is weak/empty, or as a supplement:
        print("Searching web for updated info...")
        web_result = search.run(user_input)
        full_context = f"LOCAL PDF DATA:\n{context}\n\nWEB SEARCH DATA:\n{web_result}"

        # 3. Generate Response
        formatted_prompt = prompt.format(context=full_context, input=user_input)
        
        try:
            response = llm.invoke(formatted_prompt)
            return response, sources
        except Exception as e:
            return f"Error: {e}", []

    print("\n--- Medical Helper Ready (FR/AR/EN) ---")
    while True:
        user_input = input("Posez votre question / Ask / اسأل سؤالك: ")
        if user_input.lower() == 'exit': break
        
        response, sources = rag_chain(user_input)
        
        print(f"\nAI Response: {response}")
        print(f"\nSources used: {', '.join([Path(s).name for s in sources])} & Web Search")
        print("-" * 30)

if __name__ == "__main__":
    main()