# utils/data_loader.py
"""
Chargement et extraction de contenu à partir de fichiers hétérogènes
(PDF, DOCX, TXT, CSV, XLSX) pour alimenter un pipeline RAG.

- Texte PDF      : pdfplumber (meilleure qualité que PyPDF2)
- Tableaux PDF   : tabula-py (avec traçage de la page d'origine)
- Autres formats : pandas / python-docx / lecture brute
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any, Optional, TypedDict

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

class ResultatExtraction(TypedDict):
    """Structure standardisée retournée par tous les extracteurs."""
    texte: str
    pages: list[dict]           # [{"page": 1, "texte": "..."}, ...]
    tableaux: list              # list[pd.DataFrame] avec .attrs["page"]
    source: str
    type: str                   # "pdf", "docx", "txt", "csv", "excel", "image"


def _resultat_vide(source: str, type_fichier: str) -> ResultatExtraction:
    return {
        "texte": "",
        "pages": [],
        "tableaux": [],
        "source": source,
        "type": type_fichier,
    }


# ---------------------------------------------------------------------------
# Sérialisation des tableaux pour le RAG
# ---------------------------------------------------------------------------

def _serialiser_tableau(df, format: str = "markdown") -> str:
    """
    Convertit un DataFrame en texte exploitable par un embedding.
    Les colonnes multi-niveaux  sont aplaties.
    """
    import pandas as pd

    if df.empty:
        return ""

    # Aplatir les colonnes multi-niveaux
    df = df.copy()
    df.columns = [
        " ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c)
        for c in df.columns
    ]
    # Nettoyer les cellules
    df = df.replace(r"\n", " ", regex=True).replace(r"\s+", " ", regex=True)

    if format == "markdown":
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)
    elif format == "csv":
        return df.to_csv(index=False)
    elif format == "phrases":
        entetes = df.columns.tolist()
        return "\n".join(
            " | ".join(f"{col} : {val}" for col, val in zip(entetes, row))
            for _, row in df.iterrows()
        )
    else:
        raise ValueError(f"Format inconnu : {format}")


# ---------------------------------------------------------------------------
# Extraction PDF : pdfplumber (texte) + tabula (tableaux)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    file_path: str,
    extraire_tableaux: bool = True,
    pages: Optional[list[int]] = None,
    mot_de_passe: Optional[str] = None,
) -> ResultatExtraction:
    """Extrait texte (pdfplumber) et tableaux (tabula) d'un PDF."""
    resultat = _resultat_vide(file_path, "pdf")

    # --- 1. Texte via pdfplumber ---
    try:
        import pdfplumber

        with pdfplumber.open(file_path, password=mot_de_passe) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                if pages and i not in pages:
                    continue
                texte_page = page.extract_text() or ""
                resultat["pages"].append({"page": i, "texte": texte_page})

        resultat["texte"] = "\n\n".join(p["texte"] for p in resultat["pages"])
        logger.info(
            "Texte extrait (pdfplumber) : %s (%d caractères, %d pages)",
            file_path, len(resultat["texte"]), len(resultat["pages"]),
        )
    except Exception as e:
        logger.error("Erreur extraction texte PDF %s : %s", file_path, e)
        return resultat

    # --- 2. Tableaux via tabula ---
    if extraire_tableaux:
        _extraire_tableaux_tabula(file_path, resultat, pages, mot_de_passe)

    return resultat


def _extraire_tableaux_tabula(
    file_path: str,
    resultat: ResultatExtraction,
    pages: Optional[list[int]],
    mot_de_passe: Optional[str],
) -> None:
    """Extraction tabula, page par page, pour tracer l'origine."""
    try:
        from tabula.io import read_pdf
    except ImportError:
        logger.error("[tabula] Non installé. Lancez : pip install tabula-py")
        return

    num_pages = pages or [p["page"] for p in resultat["pages"]]

    for num_page in num_pages:
        try:
            dfs = read_pdf(
                file_path,
                pages=str(num_page),
                password=mot_de_passe,
                multiple_tables=True,
                lattice=True,               # bordures visibles
                pandas_options={"header": 0},
            )
            for idx, df in enumerate(dfs):
                if df.empty:
                    continue
                df.attrs["page"] = num_page
                df.attrs["table_index"] = idx
                df.attrs["source"] = file_path
                resultat["tableaux"].append(df)
        except Exception as e:
            logger.warning(
                "[tabula] Échec page %s de %s : %s", num_page, file_path, e
            )

    logger.info(
        "Tableaux extraits (tabula) : %s (%d tableaux)",
        file_path, len(resultat["tableaux"]),
    )

def extract_text_from_png(
    file_path: str,
    langue: str = "fra+eng",
) -> ResultatExtraction:
    """
    Extrait le texte d'une image PNG via OCR (Tesseract).

    Args:
        file_path: Chemin du fichier PNG.
        langue: Langues Tesseract, ex. "fra+eng", "eng", "fra".

    Returns:
        ResultatExtraction avec type="image".
    """
    resultat = _resultat_vide(file_path, "image")
    try:
        import pytesseract
        from PIL import Image

        with Image.open(file_path) as img:
            # Prétraitement : niveaux de gris améliorent souvent l'OCR
            img = img.convert("L")
            texte = pytesseract.image_to_string(img, lang=langue)

        resultat["texte"] = texte
        resultat["pages"] = [{"page": 1, "texte": texte}]
        logger.info(
            "Texte extrait de PNG (OCR) : %s (%d caractères)",
            file_path, len(texte),
        )
    except ImportError as e:
        logger.error(
            "Dépendance manquante pour l'OCR PNG (%s). "
            "Lancez : pip install pytesseract pillow", e,
        )
    except Exception as e:
        logger.error("Erreur extraction PNG %s : %s", file_path, e)
    return resultat


# ---------------------------------------------------------------------------
# Extraction DOCX / TXT / CSV / Excel
# ---------------------------------------------------------------------------

def extract_text_from_docx(file_path: str) -> ResultatExtraction:
    resultat = _resultat_vide(file_path, "docx")
    try:
        import docx

        doc = docx.Document(file_path)
        paragraphes = [p.text for p in doc.paragraphs if p.text.strip()]
        resultat["texte"] = "\n".join(paragraphes)
        resultat["pages"] = [{"page": 1, "texte": resultat["texte"]}]
        logger.info(
            "Texte extrait de DOCX : %s (%d caractères)",
            file_path, len(resultat["texte"]),
        )
    except Exception as e:
        logger.error("Erreur extraction DOCX %s : %s", file_path, e)
    return resultat


def extract_text_from_txt(file_path: str) -> ResultatExtraction:
    resultat = _resultat_vide(file_path, "txt")
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            texte = f.read()
        resultat["texte"] = texte
        resultat["pages"] = [{"page": 1, "texte": texte}]
        logger.info(
            "Texte extrait de TXT : %s (%d caractères)", file_path, len(texte)
        )
    except Exception as e:
        logger.error("Erreur extraction TXT %s : %s", file_path, e)
    return resultat


def extract_text_from_csv(file_path: str) -> ResultatExtraction:
    resultat = _resultat_vide(file_path, "csv")
    try:
        import pandas as pd

        df = _lire_csv_robuste(file_path)
        if df is None:
            return resultat

        resultat["texte"] = df.to_string()
        resultat["pages"] = [{"page": 1, "texte": resultat["texte"]}]
        resultat["tableaux"] = [] #[df] #[]
        df.attrs["page"] = 1
        df.attrs["source"] = file_path
        #resultat["texte"] =[]  # C'est le tableau qui nous interesse


        logger.info(
            "Texte extrait de CSV : %s (%d caractères)", file_path, len(resultat["texte"])
        )
    except ImportError:
        logger.warning("Pandas non installé. Impossible de lire les CSV.")
    except Exception as e:
        logger.error("Erreur extraction CSV %s : %s", file_path, e)
    return resultat


def _lire_csv_robuste(file_path: str):
    """Tente plusieurs encodages et séparateurs avant d'abandonner."""
    import pandas as pd

    tentatives = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin1", "sep": ","},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "latin1", "sep": ";"},
        {"encoding": "utf-8", "sep": "\t"},
    ]
    for params in tentatives:
        try:
            return pd.read_csv(file_path, **params)
        except Exception:
            continue
    logger.error("Impossible de lire le CSV %s avec les encodages testés.", file_path)
    return None


def extract_text_from_excel(file_path: str) -> ResultatExtraction:
    resultat = _resultat_vide(file_path, "excel")
    try:
        import pandas as pd

        feuilles = pd.read_excel(file_path, sheet_name=None)
        blocs_texte: list[str] = []

        for nom_feuille, df in feuilles.items():
            blocs_texte.append(f"--- Feuille : {nom_feuille} ---\n{df.to_string()}")
            df.attrs["page"] = nom_feuille
            df.attrs["source"] = file_path
            resultat["tableaux"]= []
            #resultat["tableaux"].append(df)

        resultat["texte"] = "\n\n".join(blocs_texte)
        resultat["pages"] = [{"page": 1, "texte": resultat["texte"]}]
        logger.info(
            "Texte extrait de Excel : %s (%d caractères, %d feuilles)",
            file_path, len(resultat["texte"]), len(feuilles),
        )
        #resultat["texte"] = [] # C'est le tableau qui nous interesse.
    except ImportError:
        logger.warning("Pandas ou openpyxl non installé. Impossible de lire les Excel.")
    except Exception as e:
        logger.error("Erreur extraction Excel %s : %s", file_path, e)
    return resultat


# ---------------------------------------------------------------------------
# Téléchargement / extraction de ZIP
# ---------------------------------------------------------------------------

def download_and_extract_zip(url: str, output_dir: str) -> bool:
    """Télécharge un ZIP depuis une URL et l'extrait dans output_dir."""
    if not url:
        logger.warning("Aucune URL fournie pour le téléchargement.")
        return False

    try:
        logger.info("Téléchargement depuis %s...", url)
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            logger.info("Extraction dans %s...", output_dir)
            z.extractall(output_dir)

        logger.info("Téléchargement et extraction terminés.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error("Erreur de téléchargement : %s", e)
    except zipfile.BadZipFile:
        logger.error("Le fichier téléchargé n'est pas un ZIP valide.")
    except Exception as e:
        logger.error("Erreur inattendue : %s", e)
    return False


# ---------------------------------------------------------------------------
# Dispatcher par extension
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".pdf": lambda p: extract_text_from_pdf(p, extraire_tableaux=True),
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
    ".csv": extract_text_from_csv,
    ".xlsx": extract_text_from_excel,
    ".xls": extract_text_from_excel,
    ".png": extract_text_from_png,
}


def extraire_fichier(file_path: str) -> Optional[ResultatExtraction]:
    """Détecte le type et appelle l'extracteur approprié."""
    chemin = Path(file_path)
    if not chemin.exists():
        logger.error("Fichier introuvable : %s", file_path)
        return None

    extracteur = _EXT_MAP.get(chemin.suffix.lower())
    if extracteur is None:
        logger.warning("Extension non supportée : %s", chemin.suffix)
        return None

    return extracteur(str(chemin))


# ---------------------------------------------------------------------------
# Chargement récursif d'un répertoire
# ---------------------------------------------------------------------------

def load_and_parse_files(
    input_dir: str,
    extraire_tableaux_pdf: bool = True,
) -> list[dict[str, Any]]:
    """
    Parcourt récursivement `input_dir` et retourne une liste de documents
    au format {"page_content": str, "metadata": dict}.

    Deux types de documents sont produits pour les PDF :
      - un par page, avec metadata["type"] = "text"
      - un par tableau, avec metadata["type"] = "table"

    Args:
        input_dir: Répertoire source.
        extraire_tableaux_pdf: Si False, désactive tabula (plus rapide).

    Returns:
        list[dict] : documents prêts pour le chunking / embedding.
    """
    documents: list[dict[str, Any]] = []
    input_path = Path(input_dir)

    if not input_path.is_dir():
        logger.error("Le répertoire d'entrée '%s' n'existe pas.", input_dir)
        return []

    logger.info("Parcours du répertoire source : %s", input_dir)

    for file_path in sorted(input_path.rglob("*")):
        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(input_path)
        source_folder = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        ext = file_path.suffix.lower()

        # Sélection de l'extracteur
        if ext == ".pdf":
            resultat = extract_text_from_pdf(
                str(file_path),
                extraire_tableaux=extraire_tableaux_pdf,
            )
        elif ext in (".docx", ".txt", ".md", ".csv", ".xlsx", ".xls",".png"):
            resultat = extraire_fichier(str(file_path))
        else:
            logger.warning("Type non supporté ignoré : %s", relative_path)
            continue

        if not resultat:
            continue

        # --- Documents texte (un par page) ---
        for p in resultat["pages"]:
            if not p["texte"].strip():
                continue
            documents.append({
                "page_content": p["texte"],
                "metadata": {
                    "source": str(relative_path),
                    "filename": file_path.name,
                    "category": source_folder,
                    "full_path": str(file_path.resolve()),
                    "page": p["page"],
                    "type": "text",
                    "format": resultat["type"],
                },
            })

        # --- Documents tableaux (un par tableau) ---
        for df in resultat["tableaux"]:
            contenu = _serialiser_tableau(df, format="markdown")
            if not contenu.strip():
                continue
            documents.append({
                "page_content": contenu,
                "metadata": {
                    "source": str(relative_path),
                    "filename": file_path.name,
                    "category": source_folder,
                    "full_path": str(file_path.resolve()),
                    "page": df.attrs.get("page"),
                    "table_index": df.attrs.get("table_index"),
                    "n_lignes": len(df),
                    "n_colonnes": len(df.columns),
                    "type": "table",
                    "format": resultat["type"],
                },
            })

    logger.info("%d documents chargés et parsés.", len(documents))
    return documents