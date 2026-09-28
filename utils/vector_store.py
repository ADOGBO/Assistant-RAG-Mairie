# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from mistralai.client import Mistral
#from mistralai.exceptions import MistralAPIException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document # Utilisé pour le format attendu par le splitter

from .get_api_key import get_mistral_client
from .chunk_spacy import chunking_spacy

from .config import (
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP,
    CHUNK_SIZE_TABLEAU, CHUNK_OVERLAP_TABLEAU, CHUNKER_LES_TABLEAUX,
    MODELE_SPACY,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, any]] = []
        self.mistral_client = get_mistral_client()
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(
        self,
        documents: List[Dict[str, any]],
        tools: str = "spacy",
    ) -> List[Dict[str, any]]:
        """
        Découpe les documents en chunks avec métadonnées.

        Args:
            documents: Liste de dicts {page_content, metadata}.
            tools: Méthode de découpage :
                - "recursive" : RecursiveCharacterTextSplitter de LangChain
                - "spacy"     : chunking_spacy (segmentation par phrases)

        Returns:
            Liste de dicts {id, text, metadata}.
        """
        if tools not in ("recursive", "spacy"):
            raise ValueError(f"Type de découpage inconnu : {tools}")

        logging.info(
            "Découpage de %d documents (méthode=%s, taille=%d, chevauchement=%d)...",
            len(documents), tools, CHUNK_SIZE, CHUNK_OVERLAP,
        )

        # Initialisation du splitter recursive (une seule fois, pas par doc)
        if tools == "recursive":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
            )

        all_chunks: List[Dict[str, any]] = []
        doc_counter = 0

        for doc in documents:
            langchain_doc = Document(
                page_content=doc["page_content"],
                metadata=dict(doc["metadata"]),
            )
            est_csv_excel= (langchain_doc.metadata.get("format") == "csv" or langchain_doc.metadata.get("format") == "excel")
            est_tableau = langchain_doc.metadata.get("type") == "table"

            if est_csv_excel:
                langchain_doc.metadata["type"]="table" # csv et excel doit estre traitcomme des tableaus.

            # --- Sélection du chunker selon méthode + type ---
            if tools == "recursive":
                if est_tableau:
                    # Un tableau ne doit pas être coupé par caractères :
                    # on le garde tel quel (markdown préservé).
                    chunks = [langchain_doc]
                else:
                    chunks = splitter.split_documents([langchain_doc])

            else:  # spacy
                # chunking_spacy gère lui-même texte + tableaux
                chunks = chunking_spacy(
                    [langchain_doc],
                    modele=MODELE_SPACY,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    chunk_size_tableau=CHUNK_SIZE_TABLEAU,
                    chunk_overlap_tableau=CHUNK_OVERLAP_TABLEAU,
                    longueur_min=30,
                    chunker_les_tableaux=CHUNKER_LES_TABLEAUX,
                )
            logging.info(
                "  Document '%s' (%s) découpé en %d chunks.",
                doc["metadata"].get("filename", "N/A"),
                "tableau" if est_tableau else "texte",
                len(chunks),
            )

            # --- Enrichissement des métadonnées ---
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc_counter}_{i}",
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "chunk_id_in_doc": i,
                        "chunk_total_in_doc": len(chunks),
                        "start_index": chunk.metadata.get("start_index", -1),
                        "chunk_methode": tools,
                    },
                })

            doc_counter += 1

        logging.info("Total de %d chunks créés.", len(all_chunks))
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not self.mistral_client:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings.create(
                    model=EMBEDDING_MODEL,          #Tokens:  20 000 000  Requêtes par minute et 1  par seconde 
                    inputs=texts_to_embed
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            #except MistralAPIException as e:
            #    logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
            #    logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                 # Gérer l'erreur: ici on ajoute des vecteurs nuls pour ne pas bloquer
                num_failed = len(texts_to_embed)
                if all_embeddings: # Si on a déjà des embeddings, on prend la dimension du premier
                    dim = len(all_embeddings[0])
                else: # Sinon, on ne peut pas déterminer la dimension, on saute ce lot
                     logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                     continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)

            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")
                # Gérer comme ci-dessus
                num_failed = len(texts_to_embed)
                if all_embeddings:
                    dim = len(all_embeddings[0])
                else:
                     logging.error("Impossible de déterminer la dimension des embeddings, saut du lot.")
                     continue
                logging.warning(f"Ajout de {num_failed} vecteurs nuls de dimension {dim} pour le lot échoué.")
                all_embeddings.extend([np.zeros(dim, dtype='float32')] * num_failed)


        if not all_embeddings:
             logging.error("Aucun embedding n'a pu être généré.")
             return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, any]]):
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        # 1. Découper en chunks
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        # 2. Générer les embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            # Nettoyer pour éviter un état incohérent
            self.document_chunks = []
            self.index = None
            # Supprimer les fichiers potentiellement corrompus
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
            return


        # 3. Créer l'index Faiss optimisé pour la similarité cosinus
        dimension = embeddings.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        # Créer un index pour la similarité cosinus (IndexFlatIP = produit scalaire)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. Sauvegarder l'index et les chunks
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat

        Returns:
            Liste des chunks pertinents avec leurs scores
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not self.mistral_client:
             logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
             return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            # 1. Générer l'embedding de la requête
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text] # La requête doit être une liste
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')

            # Normaliser l'embedding de la requête pour la similarité cosinus
            faiss.normalize_L2(query_embedding)

            # 2. Rechercher dans l'index Faiss
            # Pour IndexFlatIP: scores = produit scalaire (plus grand = meilleur)
            # indices: index des chunks correspondants dans self.document_chunks
            # Demander plus de résultats si un score minimum est spécifié
            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            # 3. Formater les résultats
            results = []
            if indices.size > 0: # Vérifier s'il y a des résultats
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks): # Vérifier la validité de l'index
                        chunk = self.document_chunks[idx]
                        # Convertir le score en similarité (0-1)
                        # Pour IndexFlatIP avec vecteurs normalisés, le score est déjà entre -1 et 1
                        # On le convertit en pourcentage (0-100%)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        # Filtrer les résultats en fonction du score minimum
                        # Le min_score est entre 0 et 1, mais similarity est en pourcentage (0-100)
                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity, # Score de similarité en pourcentage
                            "raw_score": raw_score, # Score brut pour débogage
                            "text": chunk["text"],
                            "metadata": chunk["metadata"] # Contient source, category, chunk_id_in_doc, start_index etc.
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            # Trier par score (similarité la plus élevée en premier)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Limiter au nombre demandé (k) si nécessaire
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        #except MistralAPIException as e:
        #    logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
        #    logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
        #    return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []