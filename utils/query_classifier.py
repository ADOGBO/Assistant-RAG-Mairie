"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from mistralai.client import Mistral

from .config import CLASSIFIER_MODEL, COMMUNE_NAME, USE_LLM_FOR_CLASSIFYING
from .get_api_key import get_groq_client

def construire_prompt_session(messages, prompt_system=None, question=None, max_messages=3):
    """
    Construit un prompt enrichi avec les segments pertinents et l'historique récent.
    """
    # Limiter le nombre de messages récents
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Si une question est fournie, rechercher les segments pertinents
    context_segments = []
    if question:
        context_segments = rechercher_segments_pertinents(question)
    
    # Création du système prompt avec le contexte
    system_prompt = "Vous êtes l'assistant virtuel de la mairie de Trifouillis-sur-Loire. "
    if context_segments:
        system_prompt += "Veuillez utiliser les informations suivantes pour répondre à la question:\n\n"
        system_prompt += "\n\n".join(context_segments)
        system_prompt += "\n\nSi les informations fournies ne sont pas suffisantes pour répondre précisément, veuillez l'indiquer."
    
    # Création des messages formatés
    formatted_messages = [ChatMessage(role="system", content=system_prompt)]
    
    # Ajout des messages récents
    for msg in recent_messages:
        formatted_messages.append(ChatMessage(role=msg["role"], content=msg["content"]))
    
    return formatted_messages

class QueryClassifier:
    """
    Classe pour classifier les requêtes et déterminer si elles nécessitent RAG
    """
    
    def __init__(self,use_llm : bool=True):
        """
        Initialise le classificateur de requêtes
        """
        self.use_llm=use_llm
        self.groq_client = get_groq_client() if use_llm else None 
        
        # Mots-clés liés à la commune qui suggèrent un besoin de RAG
        self.commune_keywords = [
            COMMUNE_NAME.lower(),
            "mairie", "commune", "ville", "municipal", "municipalité",
            "conseil", "maire", "adjoint", "élu", "service",
            "horaire", "ouverture", "fermeture", "adresse", "contact",
            "document", "formulaire", "démarche", "administrative",
            "urbanisme", "permis", "construction", "travaux",
            "école", "crèche", "garderie", "cantine", "scolaire",
            "association", "sport", "culture", "loisir", "bibliothèque",
            "événement", "manifestation", "fête", "marché",
            "transport", "bus", "circulation", "stationnement", "parking",
            "déchet", "poubelle", "recyclage", "environnement",
            "impôt", "taxe", "budget", "finance"
        ]
        
        # Questions générales qui ne nécessitent pas de RAG
        self.general_patterns = [
            r"^(bonjour|salut|hello|coucou|hey|bonsoir)[\s\.,!]*$",
            r"^(merci|thanks|thank you|je te remercie)[\s\.,!]*$",
            r"^(comment ça va|ça va|comment vas-tu|comment allez-vous)[\s\.,!?]*$",
            r"^(au revoir|bye|à bientôt|à plus tard|à la prochaine)[\s\.,!]*$",
            r"^(qui es[- ]tu|qu'es[- ]tu|que fais[- ]tu|comment fonctionnes[- ]tu|tu es quoi)[\s\?]*$",
            r"^(aide|help|sos|besoin d'aide)[\s\.,!?]*$"
        ]
    
    def needs_rag(self, query: str) -> Tuple[bool, float, str]:
        """
        Détermine si une requête nécessite RAG
        
        Args:
            query: Requête de l'utilisateur
            
        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        # Convertir la requête en minuscules pour la comparaison
        query_lower = query.lower()
        
        # 1. Vérifier les patterns de questions générales (salutations, remerciements, etc.)
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return False, 0.95, "Question générale ou salutation"
        
        # 2. Vérifier la présence de mots-clés liés à la commune
        commune_keywords_found = [kw for kw in self.commune_keywords if kw in query_lower]
        if commune_keywords_found:
            keywords_str = ", ".join(commune_keywords_found)
            return True, 0.9, f"Contient des mots-clés liés à la commune: {keywords_str}"
        
        # 3. Utiliser le LLM pour les cas ambigus
        if self.use_llm:
            return self._classify_with_llm(query)
        
        # Par défaut, utiliser RAG pour les questions longues (plus de 5 mots)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Question complexe (plus de 5 mots)"
        
        # Par défaut, ne pas utiliser RAG
        return False, 0.5, "Aucun critère spécifique détecté"
    
    def _classify_with_llm(self, query: str) -> Tuple[bool, float, str]:
        """
        Utilise le LLM pour classifier la requête
        
        Args:
            query: Requête de l'utilisateur
            
        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        try:
            system_prompt = f"""Vous êtes un classificateur de requêtes pour un assistant virtuel de la commune de {COMMUNE_NAME}.
            Vous allez recevoir des questions principalement  dans la langue français et donc vous devez répondre aussi en français.
            Votre tâche est de déterminer si une question nécessite une recherche dans une base de connaissances spécifique à la commune.

            Répondez UNIQUEMENT par "RAG" ou "DIRECT" suivi d'une brève explication:
            - "RAG" si la question porte sur des informations spécifiques à {COMMUNE_NAME} (services municipaux, événements, adresses, horaires, etc.)
            - "DIRECT" si c'est une question générale, une salutation, ou une question qui ne nécessite pas d'informations spécifiques à la commune.

            Exemples:
            Question: "Bonjour, comment ça va?"
            Réponse: DIRECT - Simple salutation

            Question: "Quels sont les horaires de la mairie?"
            Réponse: RAG - Demande d'informations spécifiques à la commune

            Question: "Qui est le maire actuel?"
            Réponse: RAG - Demande d'informations spécifiques à la commune

            Question: "Qu'est-ce que l'intelligence artificielle?"
            Réponse: DIRECT - Question générale de connaissance
            """
            
            messages = []
            messages.append({ "role" : "system", "content": system_prompt})
            messages.append({"role": "user", "content":query})
                
            
            response = self.groq_client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                messages=messages,
                temperature=0.1,  # Température basse pour des réponses cohérentes
                max_completion_tokens=2048,  # Réponse courte suffisante
                top_p=0.95,
                reasoning_effort="high", #
                stream=True,
                stop=None
            )

            raisonnement = ""
            response_text = ""
            
            for chunk in  response:
                delta = chunk.choices[0].delta
                # Contenu du raisonnement (peut être absent)
                if getattr(delta, "reasoning", None):
                    raisonnement += delta.reasoning
                # Contenu de la réponse finale
                if delta.content:
                    
                    response_text += delta.content

            response_text=response_text.strip()
            logging.info(f"Classification LLM pour '{query}': {response_text}")
            logging.info(f"Voici le raisonnement du model:{raisonnement}")
            
            # Analyser la réponse
            if response_text.startswith("RAG"):
                confidence = 0.85  # Confiance élevée dans la décision du LLM
                reason = response_text.replace("RAG - ", "").replace("RAG-", "").replace("RAG:", "").strip()
                return True, confidence, reason
            elif response_text.startswith("DIRECT"):
                confidence = 0.85
                reason = response_text.replace("DIRECT - ", "").replace("DIRECT-", "").replace("DIRECT:", "").strip()
                return False, confidence, reason
            else:
                # Réponse ambiguë, utiliser RAG par défaut
                return True, 0.6, "Classification ambiguë, utilisation de RAG par précaution"
                
        except Exception as e:
            logging.error(f"Erreur lors de la classification avec LLM: {e}")
            # En cas d'erreur, utiliser RAG par défaut
            return True, 0.5, f"Erreur de classification: {str(e)}"