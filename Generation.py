from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Charger la base de vecteurs
VECTOR_DB_DIR = "./vector" 

# Initialiser les embeddings 
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)

# Charger la base de vecteurs
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

# Configurer le modèle Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    max_output_tokens=2048  # Nombre maximum de tokens générés
)

# Créer un prompt personnalisé (optionnel)
prompt_template = """
Vous êtes un assistant intelligent et détaillé. Votre rôle est de répondre aux questions de l'utilisateur en utilisant uniquement les informations fournies dans le contexte. Voici les règles à suivre :

1. **Répondez de manière claire et structurée** :
   - Commencez par une brève introduction qui résume la réponse.
   - Ensuite, fournissez des détails et des explications supplémentaires basés sur le contexte.
   - Si nécessaire, organisez la réponse en points ou en paragraphes pour une meilleure lisibilité.

2. **Restez strictement dans le contexte** :
   - N'utilisez que les informations fournies dans le contexte pour répondre.
   - Si la question est en dehors du contexte, répondez poliment que vous ne pouvez pas répondre à cette question.

3. **Ne inventez pas de réponses** :
   - Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas.
   - Ne faites pas de suppositions ou d'extrapolations en dehors du contexte.

4. **Fournissez des explications supplémentaires** :
   - Expliquez les concepts ou les termes techniques si cela aide à clarifier la réponse.
   - Si le contexte contient des exemples ou des cas d'utilisation, mentionnez-les pour enrichir la réponse.

5. **Exemple de réponse idéale** :
   - "D'après les informations fournies, [résumé de la réponse]. Pour expliquer plus en détail, [détails et explications]. Par exemple, [exemple ou cas d'utilisation]. En conclusion, [synthèse]."

Contexte :
{context}

Question :
{question}

Réponse :
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Configurer le système de question-réponse
qa_system = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Méthode de traitement des documents
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),  # Nombre de documents à récupérer
    chain_type_kwargs={"prompt": prompt},  # Utiliser le prompt personnalisé
    return_source_documents=True  # Retourner les documents sources
)



# Créer l'application FastAPI
app = FastAPI(title="RAG Web Service", description="Un service web pour interroger un système RAG basé sur Gemini et LangChain.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autoriser tous les domaines (à adapter en production)
    allow_credentials=True,
    allow_methods=["*"],  # Autoriser toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autoriser tous les en-têtes
)
# Modèle Pydantic pour la requête
class QuestionRequest(BaseModel):
    question: str

# Endpoint pour poser une question
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Poser la question au système
        result = qa_system({"query": request.question})
        
        # Retourner la réponse et les sources
        return {
            # "question": request.question,
            "answer": result["result"],
            # "sources": [
            #     {
            #         "source": doc.metadata["source"],
            #         "content": doc.page_content
            #     }
            #     for doc in result["source_documents"]
            # ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2005)