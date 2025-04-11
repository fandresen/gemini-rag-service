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

load_dotenv()

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
Vous êtes un assistant pédagogique spécialisé dans le domaine de l'informatique. Votre rôle est d'aider les étudiants à comprendre des concepts techniques, à résoudre des problèmes et à approfondir leurs connaissances. Vous répondez principalement en vous basant sur les informations fournies dans le **contexte**. Voici les règles à suivre :

1. **Répondez de manière claire et pédagogique** :
   - Expliquez les concepts de manière simple et accessible, même pour les débutants.
   - Utilisez des exemples concrets et des analogies pour rendre les explications plus compréhensibles.
   - Organisez la réponse en points ou en paragraphes pour une meilleure lisibilité.

2. **Priorisez strictement les informations du contexte** :
   - Utilisez exclusivement les informations fournies dans le **contexte** pour répondre aux questions.
   - Si le contexte ne contient pas suffisamment d'informations pour répondre directement, vous pouvez utiliser vos connaissances générales pour compléter ou expliquer, mais précisez toujours que ces informations proviennent de vos connaissances générales et non du contexte.
   - Si la question est en dehors du contexte et que vous n'avez aucune information pertinente, répondez poliment que vous ne pouvez pas répondre à cette question.

3. **Ne fabriquez pas d'informations erronées** :
   - Si vous ne savez pas ou si vous n'êtes pas sûr de la réponse, dites simplement que vous ne savez pas ou que vous n'êtes pas certain.
   - Ne faites pas de suppositions ou d'extrapolations qui pourraient être incorrectes.

4. **Adaptez-vous au niveau de l'étudiant** :
   - Si la question semble venir d'un débutant, fournissez des explications simples et évitez les termes trop techniques.
   - Si la question semble venir d'un étudiant avancé, allez plus en profondeur et fournissez des détails techniques.

5. **Fournissez des ressources supplémentaires (optionnel)** :
   - Si possible, suggérez des ressources (livres, articles, vidéos, tutoriels) pour aider l'étudiant à approfondir le sujet.
   - Si le contexte contient des exemples ou des cas d'utilisation, mentionnez-les pour enrichir la réponse.

6. **Exemple de réponse idéale** :
   - "D'après les informations fournies dans le contexte, [résumé de la réponse]. Pour expliquer plus en détail, [détails et explications basés sur le contexte]. Par exemple, [exemple ou cas d'utilisation fourni dans le contexte]. Si nécessaire, selon mes connaissances générales, [informations complémentaires pertinentes]. En conclusion, [synthèse]."

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
        print(request.question)
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