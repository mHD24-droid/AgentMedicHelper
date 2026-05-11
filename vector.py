from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_db():
    base_dir = Path(__file__).resolve().parent

    # Path where your PDFs are stored
    data_path = base_dir / "data"

    # Path where the processed database will be saved
    db_dir = base_dir / "vectorstore" / "db_faiss"
    db_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the PDFs (recursively from all subfolders)
    # Some PDFs fail text extraction due to unsupported font encodings.
    # DirectoryLoader/ PyPDFLoader can raise during parsing; to avoid a full stop,
    # we load PDFs one-by-one and skip failures.
    pdf_paths = sorted(data_path.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found under: {data_path}")

    documents = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
        except Exception as e:
            # Skip PDFs that fail during text extraction.
            print(f"[WARN] Skipping unreadable PDF: {pdf_path} ({e})")
            continue

    if not documents:
        raise RuntimeError(f"No text could be extracted from PDFs under: {data_path}")

    # 2. Split text into chunks (Medical data needs context, so we use overlap)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    if not texts:
        raise RuntimeError("Text splitting produced 0 chunks; vector DB will not be created.")



    # 3. Create Embeddings (Converts text to numbers)


    # Use a multilingual model
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

    # 4. Create and Save the Vector Store
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(str(db_dir))
    print(f"Database saved to {db_dir}")


if __name__ == "__main__":
    create_vector_db()

