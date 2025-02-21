import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration initiale
DATA_DIR = "./data"
VECTOR_DB_DIR = "./vector"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Chargement des fichiers texte
def load_documents():
    documents = []
    data_path = Path(DATA_DIR)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Le dossier {DATA_DIR} n'existe pas")
    
    for txt_file in data_path.glob("*.txt"):
        try:
            loader = TextLoader(str(txt_file), encoding='utf-8')
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = txt_file.name
            documents.extend(docs)
        except Exception as e:
            print(f"Erreur lors du chargement de {txt_file}: {str(e)}")
    
    if not documents:
        raise ValueError("Aucun document valide trouvé")
    
    return documents

# 2. Découpage des documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(documents)

# 3. Initialisation des embeddings
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

# 4. Création et stockage du vecteur
def create_vector_store(splits, embeddings):
    return Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )

# Exécution du pipeline
if __name__ == "__main__":
    # Charger les documents
    raw_docs = load_documents()
    print(f"{len(raw_docs)} document(s) chargé(s)")
    
    # Découpage
    splits = split_documents(raw_docs)
    print(f"Document découpés en {len(splits)} chunks")
    
    # Initialiser les embeddings
    embeddings = get_embeddings()
    
    # Créer la base vectorielle
    vector_db = create_vector_store(splits, embeddings)
    vector_db.persist()
    print(f"Base vectorielle sauvegardée dans {VECTOR_DB_DIR}")