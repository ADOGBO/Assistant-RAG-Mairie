from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import Literal, Union, Optional
import pandas as pd
import spacy
from spacy.language import Language
from langchain_core.documents import Document

### SEMANTIC CHUNKING WITH SPACY

_NLP_CACHE: dict[str, Language] = {}


def _get_nlp(modele: str) -> Language:
    if modele not in _NLP_CACHE:
        _NLP_CACHE[modele] = spacy.load(modele)
    return _NLP_CACHE[modele]


def _chunker_tableau(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Découpe un Document tableau par lignes, en répétant l'en-tête markdown.
    Si le tableau tient dans chunk_size, il est renvoyé tel quel.
    """
    texte = doc.page_content

    # Si le tableau tient, on ne touche à rien
    if len(texte) <= chunk_size:
        return [doc]

    lignes = texte.splitlines()

    # Trouver l'en-tête markdown (les 2 premières lignes : titre colonnes + séparateur)
    if len(lignes) >= 2 and lignes[1].lstrip().startswith("|") and "---" in lignes[1]:
        entete = lignes[:2]
        corps = lignes[2:]
    else:
        # Fallback : on considère qu'il n'y a pas d'en-tête markdown
        entete = []
        corps = lignes

    chunks: list[Document] = []
    buffer: list[str] = []
    taille_buffer = len("\n".join(entete)) + 1 if entete else 0

    def vider_buffer(idx: int):
        if not buffer:
            return
        contenu = "\n".join(entete + buffer) if entete else "\n".join(buffer)
        meta = dict(doc.metadata)
        meta["table_chunk_index"] = idx
        chunks.append(Document(page_content=contenu, metadata=meta))

    idx = 0
    for ligne in corps:
        taille_ligne = len(ligne) + 1
        if taille_buffer + taille_ligne > chunk_size and buffer:
            vider_buffer(idx)
            idx += 1
            # Chevauchement : on reprend les dernières lignes
            buffer_new: list[str] = []
            taille_new = len("\n".join(entete)) + 1 if entete else 0
            for l in reversed(buffer):
                if taille_new + len(l) + 1 > chunk_overlap:
                    break
                buffer_new.insert(0, l)
                taille_new += len(l) + 1
            buffer = buffer_new
            taille_buffer = taille_new
        buffer.append(ligne)
        taille_buffer += taille_ligne

    vider_buffer(idx)

    # Marquer le nombre total
    total = len(chunks)
    for c in chunks:
        c.metadata["table_chunk_total"] = total

    return chunks


def chunking_spacy(
    documents: Union[str, list[Document]],
    modele: str = "fr_core_news_sm",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    chunk_size_tableau: int =100000,
    chunk_overlap_tableau: int = 1500,
    longueur_min: int = 30,
    chunker_les_tableaux: bool = True,
) -> list[Document]:
    """
    Découpe une liste de Document (texte + tableaux) en chunks sémantiques.

    - Les documents de type "text" passent par spaCy (segmentation par phrases).
    - Les documents de type "table" sont préservés, ou découpés par lignes
      si `chunker_les_tableaux=True` et qu'ils dépassent chunk_size.

    Args:
        documents: Une chaîne, ou une liste de Document.
        modele: Modèle spaCy.
        chunk_size: Taille cible d'un chunk en caractères.
        chunk_overlap: Chevauchement (textes et tableaux).
        longueur_min: Taille minimale pour conserver un chunk de texte.
        chunker_les_tableaux: Si True, découpe les gros tableaux par lignes.

    Returns:
        list[Document] : chunks finaux (textes + tableaux mélangés).
    """
    # 1. Normalisation de l'entrée
    if isinstance(documents, str):
        entrees = [(documents, {"type": "text"})]
    elif isinstance(documents, list):
        entrees = [(d.page_content, dict(d.metadata)) for d in documents]
    else:
        raise TypeError("documents doit être une str ou une list[Document].")

    # 2. Séparation texte / tableau
    textes_entrees = [(t, m) for t, m in entrees if m.get("type") != "table"]
    tableaux_entrees = [(t, m) for t, m in entrees if m.get("type") == "table"]

    chunks_finaux: list[Document] = []

    # 3. Chunking des textes via spaCy
    if textes_entrees:
        nlp = _get_nlp(modele)

        for texte_source, meta_source in textes_entrees:
            if not texte_source or not texte_source.strip():
                continue

            doc = nlp(texte_source)
            phrases = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if not phrases:
                continue

            chunks_texte: list[str] = []
            buffer: list[str] = []
            taille_buffer = 0

            for phrase in phrases:
                # Phrase anormalement longue -> découpe brutale
                if len(phrase) > chunk_size:
                    if buffer:
                        chunks_texte.append(" ".join(buffer))
                        buffer, taille_buffer = [], 0
                    for i in range(0, len(phrase), chunk_size - chunk_overlap):
                        morceau = phrase[i:i + chunk_size]
                        if morceau.strip():
                            chunks_texte.append(morceau.strip())
                    continue

                if taille_buffer + len(phrase) + 1 <= chunk_size:
                    buffer.append(phrase)
                    taille_buffer += len(phrase) + 1
                else:
                    chunks_texte.append(" ".join(buffer))
                    nouveau_buffer: list[str] = []
                    taille_new = 0
                    for p in reversed(buffer):
                        if taille_new + len(p) + 1 > chunk_overlap:
                            break
                        nouveau_buffer.insert(0, p)
                        taille_new += len(p) + 1
                    nouveau_buffer.append(phrase)
                    buffer = nouveau_buffer
                    taille_buffer = sum(len(p) + 1 for p in buffer)

            if buffer:
                chunks_texte.append(" ".join(buffer))

            for i, texte_chunk in enumerate(chunks_texte):
                if len(texte_chunk.strip()) < longueur_min:
                    continue
                meta = dict(meta_source)
                meta["chunk_index"] = i
                meta["chunk_total"] = len(chunks_texte)
                meta["chunk_methode"] = "spacy"
                meta["modele_spacy"] = modele
                chunks_finaux.append(
                    Document(page_content=texte_chunk.strip(), metadata=meta)
                )

    # 4. Traitement des tableaux
    for texte_table, meta_table in tableaux_entrees:
        if not texte_table or not texte_table.strip():
            continue

        doc_table = Document(page_content=texte_table, metadata=dict(meta_table))

        if chunker_les_tableaux:
            chunks_finaux.extend(
                _chunker_tableau(doc_table, chunk_size_tableau, chunk_overlap_tableau)
            )
        else:
            # On garde le tableau tel quel
            meta_table["chunk_methode"] = "table_intact"
            chunks_finaux.append(doc_table)

    return chunks_finaux