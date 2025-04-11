import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


load_dotenv()
# Configuration initiale
DATA_DIR = "./data"
VECTOR_DB_DIR = "./vector"
PROCESSED_FILES = "./processed_files.txt"  # Fichier de suivi
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Charger la liste des fichiers déjà traités
def load_processed_files():
    if os.path.exists(PROCESSED_FILES):
        with open(PROCESSED_FILES, "r") as f:
            return set(f.read().splitlines())
    return set()

# Ajouter un fichier à la liste des fichiers traités
def mark_file_as_processed(filename):
    with open(PROCESSED_FILES, "a") as f:
        f.write(f"{filename}\n")

# Charger les nouveaux fichiers
def load_new_documents():
    documents = []
    data_path = Path(DATA_DIR)
    processed_files = load_processed_files()

    if not data_path.exists():
        raise FileNotFoundError(f"Le dossier {DATA_DIR} n'existe pas")

    for file_path in data_path.glob("**/*"):
        if file_path.suffix.lower() in (".txt", ".pdf") and file_path.name not in processed_files:
            try:
                if file_path.suffix.lower() == ".txt":
                    loader = TextLoader(str(file_path), encoding='utf-8')
                else:  # Gestion des PDF
                    loader = PyPDFLoader(str(file_path))

                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "source": file_path.name,
                        "page": doc.metadata.get("page", 1)  # Pagination 1-based pour les fichiers texte
                    })

                documents.extend(docs)
                mark_file_as_processed(file_path.name)  # Marquer le fichier comme traité
                print(f"✅ {len(docs)} pages chargées depuis {file_path.name}")

            except Exception as e:
                print(f"❌ Erreur avec {file_path.name}: {str(e)}")
                continue

    if not documents:
        print("Aucun nouveau fichier à traiter.")
    
    return documents

# Découpage des documents
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_documents(documents)

# Initialisation des embeddings
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

# Création et stockage du vecteur
def create_vector_store(splits, embeddings):
    if os.path.exists(VECTOR_DB_DIR):
        # Charger la base vectorielle existante
        vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        vector_db.add_documents(splits)  # Ajouter les nouveaux chunks
    else:
        # Créer une nouvelle base vectorielle
        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_DIR
        )
    vector_db.persist()
    return vector_db

# Exécution du pipeline
if __name__ == "__main__":
    # Charger les nouveaux documents
    raw_docs = load_new_documents()
    if not raw_docs:
        exit(0)  # Aucun nouveau fichier à traiter

    # Découpage
    splits = split_documents(raw_docs)
    print(f"Document découpés en {len(splits)} chunks")

    # Initialiser les embeddings
    embeddings = get_embeddings()

    # Créer ou mettre à jour la base vectorielle
    vector_db = create_vector_store(splits, embeddings)
    print(f"\nBase vectorielle sauvegardée dans {VECTOR_DB_DIR}")