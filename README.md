# 🏛️ Assistant RAG — Mairie de Triffouillis-sur-Loire

Assistant virtuel basé sur **Retrieval-Augmented Generation (RAG)** permettant de répondre à des questions à partir d'une base de connaissances personnalisée (documents administratifs de la mairie de Triffouillis-sur-Loire). Le pipeline combine extraction de documents PDF, indexation vectorielle, retrieval sémantique et génération de réponses contextualisées par un LLM.

## ✨ Fonctionnalités

- Extraction de texte et de tableaux depuis des documents PDF (rapports, comptes-rendus, délibérations)
- Chunking hybride : segmentation sémantique du texte (spaCy) + extraction structurée des tableaux (Tabula)
- Indexation vectorielle avec recherche par similarité (FAISS)
- Génération de réponses contextualisées via LLM
- Évaluation objective du pipeline avec le framework Ragas (fidélité, pertinence, précision du retrieval)
- Interface conversationnelle via Streamlit

## 🧱 Stack technique

| Composant | Choix |
|---|---|
| Modèle d'embedding | `mistral-embed-2312` (Mistral AI) |
| LLM de génération | `openai/gpt-oss-120b` (via Groq) |
| Store vectoriel | FAISS |
| Chunking | spaCy (texte) + Tabula (tableaux) |
| Frontend | Streamlit |
| Évaluation | Ragas (Faithfulness, Context Recall, Factual Correctness, Answer Relevancy) |
| Gestion des dépendances | uv |

## 📚 Base de connaissances

- **Source** : documents administratifs de la mairie de Triffouillis-sur-Loire
- **Corpus** : 23 documents → 113 chunks après découpage

## 🔀 Stratégie de chunking

Le texte est découpé par segmentation de phrases avec spaCy, avec regroupement en chunks de taille raisonnable. Les tableaux sont traités séparément : conservés tels quels s'ils sont de taille raisonnable, ou découpés avec Tabula lorsqu'ils sont trop volumineux pour tenir dans un seul chunk. Cette approche hybride préserve la structure des données tabulaires (budgets, délibérations chiffrées) tout en gardant une segmentation cohérente du texte narratif.

## 📊 Résultats d'évaluation (Ragas)

Le pipeline a été évalué avec [Ragas](https://docs.ragas.io) sur un jeu de questions/réponses annotées manuellement à partir du corpus. Métriques évaluées : Faithfulness, Context Recall, Factual Correctness, Answer Relevancy.

Résultats détaillés disponibles dans [`resultats_evaluation.csv`](./resultats_evaluation.csv).

| Métrique | Score |
|---|---|
| Faithfulness | à compléter |
| Context Recall | à compléter |
| Factual Correctness | à compléter |
| Answer Relevancy | à compléter |

## 🚀 Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
git clone https://github.com/<ton-user>/<ton-repo>.git
cd <ton-repo>
uv sync
```

### Variables d'environnement

Créer un fichier `.env` à la racine (non versionné) :

```
MISTRAL_API_KEY=ta_cle_mistral
GROQ_API_KEY=ta_cle_groq
```

## ▶️ Utilisation

```bash
uv run streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`.

## ⚠️ Limitations connues

- Le palier gratuit de l'API Mistral impose des limites de débit (rate limiting), pouvant occasionner des erreurs 429 en cas d'usage intensif.
- Le corpus étant volontairement restreint (23 documents), les résultats de retrieval ne sont pas représentatifs d'une base documentaire de grande échelle.
- Projet à visée démonstrative/portfolio, non destiné à un usage en production.

## 📄 Licence

MIT

## 👤 Auteur

Gnonnan Jean-Paul Adogbo — [LinkedIn](#) · [Portfolio](#)
