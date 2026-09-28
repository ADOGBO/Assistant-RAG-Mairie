# utils/config.py
import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées
    # raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed-2312"
#--------- MOdel OpenAI --------------------
CLASSIFIER_MODEL = "openai/gpt-oss-safeguard-20b" #openai/gpt-oss-20b" # gpt-oss-safeguard-120b Ou un autre modèle comme mistral-large-latest
CHAT_MODEL = "openai/gpt-oss-120b" #"openai/gpt-oss-120b"
JUDGE_MODEL=  "qwen/qwen3.8-27b" #"openai/gpt-oss-20b"
USE_LLM_FOR_CLASSIFYING=True


# --- Configuration de l'Indexation ---
# INPUT_DATA_URL = os.getenv("INPUT_DATA_URL") # Décommentez si vous utilisez une URL
INPUT_DIR =  "data"               # Dossier pour les données sources après extraction
VECTOR_DB_DIR = "vector_db"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
CHUNK_SIZE_TABLEAU = 100000         # Taille des chunks pour les tableaux (en caractères)
CHUNK_OVERLAP_TABLEAU = 1500        # Chevauchement pour les tableaux (en caractères)
CHUNKER_LES_TABLEAUX = True          # Découper les tableaux en chunks si True, sinon les conserver entiers
MODELE_SPACY = "fr_core_news_sm"     # Modèle spaCy pour le découpage sémantique
EMBEDDING_BATCH_SIZE = 128           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                        # Nombre de documents à récupérer par défaut
MAX_MESSAGES=3

# --- Configuration de la Base de Données ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "Assistant RAG"
COMMUNE_NAME = "Triffouillis-sur-Loire" # Nom à personnaliser dans l'interface