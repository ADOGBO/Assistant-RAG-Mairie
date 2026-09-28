# 🏛️ Assistant RAG — Mairie de Triffouillis-sur-Loire

Assistant virtuel basé sur **Retrieval-Augmented Generation (RAG)** permettant de répondre à des questions à partir d'une base de connaissances personnalisée (documents administratifs de la mairie de Triffouillis-sur-Loire). Le pipeline combine extraction de documents PDF, indexation vectorielle, retrieval sémantique et génération de réponses contextualisées par un LLM. 

[Démo](https://assistant-rag-mairie-4hkgbjxrcgzs7vzwoyxl56.streamlit.app/)

## ✨ Fonctionnalités

- Extraction de texte et de tableaux depuis des documents PDF (rapports, comptes-rendus, délibérations)
- Chunking hybride : segmentation sémantique du texte (spaCy) + extraction structurée des tableaux (Tabula)
- Indexation vectorielle avec recherche par similarité (FAISS)
- Classification de requêtes (savoir si la requête utilisateur à besoin de RAG ou une requête hors contexte) via LLM
- Génération de réponses contextualisées via LLM
- Évaluation objective du pipeline avec le framework Ragas (fidélité, pertinence, précision du retrieval)
- Interface conversationnelle via Streamlit

## 🧱 Stack technique

| Composant | Choix |
|---|---|
| Modèle d'embedding | `mistral-embed-2312` (Mistral AI) |
| LLM de génération | `openai/gpt-oss-120b` (via Groq) |
| LLM de classification | `openai/gpt-oss-safeguard-20b` (via Groq) |
| LLM d'évaluation | `qwen/qwen3.8-27b` (via Groq) |
| Store vectoriel | FAISS |
| Chunking | spaCy (texte) + Tabula (tableaux) |
| Frontend | Streamlit |
| Évaluation | Ragas (Faithfulness, Context Recall, Factual Correctness) |
| Gestion des dépendances | uv |

## 📚 Base de connaissances

- **Source** : documents administratifs de la mairie de Triffouillis-sur-Loire
- **Corpus** : 23 documents → 113 chunks après découpage

## 🔀 Stratégie de chunking

Le texte est découpé par segmentation de phrases avec spaCy, avec regroupement en chunks de taille raisonnable. Les tableaux sont traités séparément : conservés tels quels s'ils sont de taille raisonnable, ou découpés avec Tabula lorsqu'ils sont trop volumineux pour tenir dans un seul chunk. Cette approche hybride préserve la structure des données tabulaires (budgets, délibérations chiffrées) tout en gardant une segmentation cohérente du texte narratif.

## 📊 Résultats d'évaluation (Ragas)

Les modèles LLM de génération ont été comparées grace à [Chatbot Arena](https://arena.ai/?leaderboard=). Le pipeline a été évalué avec [Ragas](https://docs.ragas.io) sur un jeu de questions/réponses annotées manuellement à partir du corpus. Métriques évaluées : Faithfulness, Context Recall, Factual Correctness, Answer Relevancy.

Résultats détaillés disponibles dans [`resultat_eval.csv`](./resultat_eval.csv).

| Métrique | Score |
|---|---|
| Faithfulness | 1 |
| Context Recall |1|
| Factual Correctness | 0.8|

## 🚀 Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
git clone https://github.com/ADOGBO/Assistant-RAG-Mairie.git
cd Assistant-RAG-Mairie
uv sync
```

### Variables d'environnement

Créer un fichier `.env` à la racine (non versionné) :

```
MISTRAL_API_KEY=ta_cle_mistral
GROQ_API_KEY=ta_cle_groq
```
## 📁 Structure du projet

\`\`\`
projet/
├── chat.py                     # Application Streamlit principale
├── indexer.py                  # Script d'indexation des documents
├── eval_stock.py               # Script d'évaluation du système RAG
├── pyproject.toml              # Dépendances gérées par uv
├── data/                       # Documents sources
├── vector_db/                  # Index FAISS et chunks vectorisés
├── database/                   # Base SQLite des interactions
└── utils/                      # Modules utilitaires
    ├── config.py               # Configuration de l'application
    ├── database.py             # Gestion de la base de données
    ├── query_classifier.py     # Classification des requêtes
    ├── chunk_spacy.py          # Découpage des documents avec spaCy
    ├── data_loader.py          # Chargement et extraction de contenu
    ├── eval_ragas.py           # Évaluation RAG avec RAGAS
    ├── get_api_key.py          # Gestion des clés API
    └── vector_store.py         # Gestion de l'index vectoriel
\`\`\`
    
    


## ▶️ Utilisation
1. Ajouter des documents

Placez vos documents dans le dossier data/. Les formats supportés sont :

-PDF
-TXT
-DOCX
-CSV
-JSON
-PNG
Vous pouvez organiser vos documents dans des sous-dossiers pour une meilleure organisation.

2. Indexer les documents

Exécutez le script d'indexation pour traiter les documents et créer l'index FAISS :
```bash
python indexer.py
```
Ce script va :

1. Charger les documents depuis le dossier inputs/
2. Découper les documents en chunks
3. Générer des embeddings avec Mistral
4. Créer un index FAISS pour la recherche sémantique
5. Sauvegarder l'index et les chunks dans le dossier vector_db/

3. Lancer l'application
```bash
uv run streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

## Fonctionnalités principales

### Classification des requêtes

L'application détermine automatiquement si une question nécessite une recherche RAG ou si une réponse directe du modèle OpenAI est suffisante. Cela permet d'optimiser les performances et la pertinence des réponses. 

### Paramètres personnalisables

Dans la barre latérale, vous pouvez ajuster :

Le modèle OpenAI (Small ou Large)
Le nombre de documents à récupérer (1-20)
Le score minimum de similarité (0-100%)
### Feedback et analyse

L'application enregistre les interactions et les feedbacks des utilisateurs. 

## ⚠️ Limitations connues

- Le palier gratuit de l'API Mistral impose des limites de débit (rate limiting), pouvant occasionner des erreurs 429 en cas d'usage intensif.
- Le corpus étant volontairement restreint (23 documents), les résultats de retrieval ne sont pas représentatifs d'une base documentaire de grande échelle.
- Projet à visée démonstrative/portfolio, non destiné à un usage en production.

## Modules principaux

### utils/vector_store.py

Gère l'index vectoriel FAISS et la recherche sémantique :

-Chargement et découpage des documents
-Génération des embeddings avec Mistral
-Création et interrogation de l'index FAISS

### utils/query_classifier.py

Détermine si une requête nécessite une recherche RAG :

-Analyse des mots-clés
-Classification avec le modèle Mistral
-Détection des questions spécifiques vs générales

### utils/database.py

Gère la base de données SQLite pour les interactions :

-Enregistrement des questions et réponses
-Stockage des feedbacks utilisateurs
-Récupération des statistiques

## 👤 Auteur

Gnonnan Jean-Paul Adogbo — [LinkedIn](https://www.linkedin.com/in/gnonnan-jean-paul-adogbo-phd-269213223/) · [Portfolio](https://assistant-rag-mairie-4hkgbjxrcgzs7vzwoyxl56.streamlit.app/)
