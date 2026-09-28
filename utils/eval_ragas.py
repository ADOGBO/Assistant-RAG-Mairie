

# utils/ragas_eval.py
"""
Évaluation RAG avec RAGAS.

Supporte plusieurs juges LLM (Anthropic, OpenAI, Mistral/Ollama, Groq).
"""

"""
Évaluation RAG avec RAGAS.

- Juge : Anthropic, OpenAI ou Mistral (via LangChain, pas de langchain-groq).
- Embeddings : Mistral (via get_mistral_client()).
- Génération RAG : client Groq natif.
"""

import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
import numpy as np
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset, evaluate
#from ragas.embeddings import LangchainEmbeddingsWrapper
#from ragas.llms import LangchainLLMWrapper
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import OpenAI
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from utils.config import COMMUNE_NAME, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,JUDGE_MODEL, SEARCH_K, CHAT_MODEL  
from utils.get_api_key import get_groq_client, get_mistral_client,get_mistral_client_via_OpenAI, get_groq_client_via_OpenAI
from utils.vector_store import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_judge(which_factory: str="Groq", model_name: Optional[str] = None):
    """Construit le LLM juge selon le provider demandé."""
    if which_factory == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name or "claude-3-5-sonnet-20241022",
            temperature=0,
        )
    elif which_factory == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "gpt-4o",
            temperature=0,
        )
    elif which_factory == "Groq":
        # Utilise le client natif via get_groq_client()
        client=get_groq_client_via_OpenAI()

        return llm_factory(
               model_name or  "qwen/qwen3.8-27b",
                provider="openai",      # <-- Important : utilise l'adaptateur OpenAI
                client=client,
                temperature=0.1,
                max_completion_tokens=1024, #2048
                top_p=0.95,
                reasoning_effort="default",
                stream=False,
                stop=None
            )

    elif which_factory == "Mistral":

        client=get_mistral_client_via_OpenAI()
        return llm_factory(
               model_name or  "mistral-large-latest",
                provider="openai",      # <-- Important : utilise l'adaptateur OpenAI
                client=client,
            )

        
    else:
        raise ValueError(
            f"which_factory inconnu : {which_factory}. "
            f"Choisir parmi 'Anthropic', 'OpenAI', 'Groq'."
        )


from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper


def _build_embeddings_mistral(which_embedding:str="HF") :
    """Embeddings Mistral via get_mistral_client()."""
    if which_embedding=="Mistral":
        client = get_mistral_client_via_OpenAI()
        #api_key = client.api_key
        return OpenAIEmbeddings(
                client=client,
                model=EMBEDDING_MODEL
            )
    
    elif which_embedding =="HF":
        from ragas.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model="BAAI/bge-m3",
            use_api=False   # False = modèle local via sentence-transformers, True = API HuggingFace hébergée
        )



# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class RagasEval:
    def __init__(
        self,
        which_factory: str = "Groq",
        judge_model: Optional[str] = JUDGE_MODEL,
        num_docs: int = SEARCH_K,
        min_score: float = 0.75,
        selected_model_gen: str = CHAT_MODEL ,
        batch_size: int = 1,
    ):
        self.num_docs = num_docs
        self.min_score = min_score
        self.selected_model_gen = selected_model_gen
        self.batch_size = batch_size

        self.vector_store = VectorStoreManager()
        self.client_llm = get_groq_client()

        # Juge RAGAS
        self.llm_judge =_build_judge(which_factory, judge_model)

        
        # Embeddings Mistral
        self.embeddings = None #_build_embeddings_mistral()

        self.dataset: Optional[list[dict]] = None

    # -----------------------------------------------------------------
    # Prompt système pour la génération
    # -----------------------------------------------------------------

    def system_prompt_llm(self, context_str: str) -> str:
        return (
            f"Vous êtes un assistant virtuel pour {COMMUNE_NAME}. "
            "Vous recevrez des instructions en français et vous avez "
            "l'obligation de répondre aussi en français.\n"
            "Répondez à la question de l'utilisateur en vous basant "
            "UNIQUEMENT sur le contexte fourni ci-dessous.\n"
            "Si l'information n'est pas dans le contexte, dites que vous "
            "ne savez pas ou que l'information n'est pas disponible dans "
            "les documents fournis.\n"
            "Soyez concis et précis. Citez vos sources si possible.\n\n"
            f"Contexte fourni:\n---\n{context_str}\n---\n"
        )

    def message_for_llmGen(self, prompt: str, context_str: str) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt_llm(context_str)},
            {"role": "user", "content": prompt},
        ]

    # -----------------------------------------------------------------
    # Retrieval + génération
    # -----------------------------------------------------------------

    def _retrieve(self, query: str) -> tuple[str, list[str]]:
        """Retourne (context_str_pour_prompt, liste_contextes_pour_RAGAS)."""
        docs = self.vector_store.search(
            query, k=self.num_docs, min_score=self.min_score
        )

        if not docs:
            logger.warning("Aucun document trouvé pour : %s", query)
            return "Aucun contexte disponible.", [""]

        context_str = "\n\n---\n\n".join(
            f"Source: {d['metadata'].get('source', 'Inconnue')} "
            f"(Score: {d['score']:.4f})\nContenu: {d['text']}"
            for d in docs
        )
        contextes = [d["text"] for d in docs]
        return context_str, contextes

    def _generate(self, query: str, context_str: str) -> str:
        """Appelle le LLM Groq de génération."""
        try:
            response = self.client_llm.chat.completions.create(
                model=self.selected_model_gen,
                messages=self.message_for_llmGen(query, context_str),
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
                reasoning_effort="medium",
                stream=False,
                stop=None,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Échec génération pour '%s' : %s", query, e)
            return ""

    def create_data_rag(
        self,
        sample_queries: list[str],
        expected_responses: list[str],
    ) -> list[dict]:
        """Exécute le RAG complet pour chaque question."""
        if len(sample_queries) != len(expected_responses):
            raise ValueError(
                "sample_queries et expected_responses doivent avoir "
                "la même longueur."
            )

        dataset_ = []
        for i, (query, reference) in enumerate(
            zip(sample_queries, expected_responses), start=1
        ):
            logger.info(
                "Évaluation %d/%d : %s", i, len(sample_queries), query
            )

            context_str, contextes = self._retrieve(query)
            reponse = self._generate(query, context_str)

            dataset_.append({
                "question": query,
                "answer": reponse,
                "contexts": contextes,
                "ground_truth": reference,
            })

        self.dataset = dataset_
        return dataset_

    # -----------------------------------------------------------------
    # Évaluation RAGAS
    # -----------------------------------------------------------------

    def eval_rag(self, dataset: Optional[list[dict]] = None):
        """Lance l'évaluation RAGAS."""
        data = dataset or self.dataset
        if not data:
            raise ValueError(
                "Aucun dataset. Appelez create_data_rag() d'abord, ou "
                "passez un dataset à eval_rag()."
            )

        evaluation_dataset = EvaluationDataset.from_list(data)

        # 2. Instancier les métriques avec le LLM juge (et les embeddings si nécessaire)
        

        from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy


        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
            llm=self.llm_judge,#llmj,#self.llm_judge,
            embeddings= None, #self.embeddings, #embedder, #None, #self.embeddings,
            show_progress=True
        )
        #print(result)
        return result

    





