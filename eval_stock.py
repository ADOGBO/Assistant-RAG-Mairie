

"""
Évaluation incrémentale d'une question avec RAGAS et stockage en CSV.

Usage :
    evaluer_et_sauvegarder(
        metrics=["faithfulness"],
        question="Quels sont les horaires ?",
        contexts=["La mairie ouvre à 9h..."],
        ground_truth="Du lundi au vendredi de 9h à 12h.",
        answer="La mairie ouvre à 9h du lundi au vendredi.",
        llm_judge=judge,
        embeddings=embeddings,
    )
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# Ordre fixe des colonnes du CSV de détail
_COLONNES = [
    "question",
    "response",
    "reference",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "score_moyen",
    "flag",
]

# Métriques reconnues (doivent correspondre aux noms RAGAS)
_METRIQUES_VALIDES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _est_vide(valeur) -> bool:
    """Vérifie si une valeur de cellule est manquante (NaN, NA, vide)."""
    if valeur is None:
        return True
    try:
        return bool(pd.isna(valeur))
    except (TypeError, ValueError):
        return False

def _construire_metriques(
    noms: list[str],
    llm_judge,
    embeddings,
) -> list:
    """Construit les instances des métriques RAGAS à partir de leurs noms."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    factory = {
        "faithfulness": lambda: Faithfulness(llm=llm_judge),
        "answer_relevancy": lambda: AnswerRelevancy(
            llm=llm_judge, embeddings=embeddings
        ),
        "context_precision": lambda: ContextPrecision(llm=llm_judge),
        "context_recall": lambda: ContextRecall(llm=llm_judge),
    }
    return [factory[nom]() for nom in noms]




def _normaliser_contextes(contexts: Union[str, list[str]]) -> list[str]:
    """Accepte une chaîne unique ou une liste."""
    if isinstance(contexts, str):
        return [contexts]
    return list(contexts)


def _extraire_scores(row: pd.Series, noms: list[str]) -> dict:
    """Retourne un dict {métrique: score} pour les métriques demandées."""
    return {nom: row.get(nom) for nom in noms}


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def evaluer_et_sauvegarder(
    num_question:int,
    sujet,
    dataset=None,
    which_metrics:str= "faithAndAnswer",
    csv_path: str = "detail_ragas.csv",
    overwrite: bool = False,
    tronquer_a: int = 200,
    batch_size: int = 1,
) -> dict:
    """
    Évalue une question avec les métriques demandées et met à jour un CSV de détail.

    - Si le CSV n'existe pas, il est créé avec toutes les colonnes.
    - Si la question existe déjà, la ligne est mise à jour.
    - Par défaut, les métriques déjà calculées ne sont pas recalculées
      (mettre `overwrite=True` pour forcer).

    Args:
        metrics: nom(s) de métrique(s) : "faithfulness", "answer_relevancy",
                 "context_precision", "context_recall". Chaîne ou liste.
        question: la question posée à l'utilisateur.
        contexts: contexte(s) récupéré(s) (str ou list[str]).
        ground_truth: la réponse de référence.
        answer: la réponse générée par le LLM.
        llm_judge: LLM juge RAGAS (InstructorBaseRagasLLM).
        embeddings: embeddings (obligatoire pour "answer_relevancy").
        csv_path: chemin du CSV de détail.
        overwrite: si True, recalcule même les métriques déjà présentes.
        tronquer_a: longueur max pour les colonnes texte (question, response…).
        batch_size: passé à ragas.evaluate (1 = sûr pour les rate limits).

    Returns:
        dict {métrique: score} pour les métriques demandées.
    """
    embeddings=sujet.embeddings
    llm_judge=sujet.llm_judge

    if dataset is not None:
        question=dataset[num_question]['question']
        contexts=dataset[num_question]['contexts']
        answer=dataset[num_question]['answer']
        ground_truth=dataset[num_question]['ground_truth']
    
    elif sujet.dataset is not None:
        question=sujet.dataset[num_question]['question']
        contexts=sujet.dataset[num_question]['contexts']
        answer=sujet.dataset[num_question]['answer']
        ground_truth=sujet.dataset[num_question]['ground_truth']
    # --- 1. Normalisation des métriques ---
    else:
        
        raise ValueError("Aucun dataset fourni.")

    """if isinstance(metrics, str):
            metrics = [metrics]
    metrics = list(metrics)
    
    inconnues = [m for m in metrics if m not in _METRIQUES_VALIDES]
    if inconnues:
        raise ValueError(
            f"Métrique(s) inconnue(s) : {inconnues}. "
            f"Choisir parmi {_METRIQUES_VALIDES}."
        )
            
    
        if "answer_relevancy" in metrics and embeddings is None:
            raise ValueError(
                "'answer_relevancy' nécessite un objet `embeddings`."
            )
"""
    # --- 2. Chargement ou création du CSV ---
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    if csv_file.exists():
        df = pd.read_csv(csv_file)
        # S'assurer que toutes les colonnes attendues existent
        for col in _COLONNES:
            if col not in df.columns:
                df[col] = pd.NA
        # Réordonner les colonnes (en gardant les colonnes inconnues à la fin)
        cols_extra = [c for c in df.columns if c not in _COLONNES]
        df = df[_COLONNES + cols_extra]
    else:
        df = pd.DataFrame(columns=_COLONNES)

    # --- 3. Localiser ou créer la ligne ---
    masque = df["question"].astype(str) == question
    idx_existant = df.index[masque]

    if len(idx_existant) > 0:
        row_idx = idx_existant[0]
        logger.info("Question déjà présente (ligne %s), mise à jour.", row_idx)
    else:
        nouvelle_ligne = {c: pd.NA for c in _COLONNES}
        nouvelle_ligne["question"] = question
        nouvelle_ligne["response"] = answer[:tronquer_a] if answer else ""
        nouvelle_ligne["reference"] = (
            ground_truth[:tronquer_a] if ground_truth else ""
        )
        df.loc[len(df)] = nouvelle_ligne
        row_idx = df.index[-1]
        logger.info("Nouvelle question ajoutée (ligne %s).", row_idx)

    # --- 4. Déterminer les métriques à calculer --- ["faith","answer","contextRecall","factualCorrectness"]
    if which_metrics=="faith":
        metrics_a_calculer=["faithfulness"]
    elif which_metrics=="answer":
        metrics_a_calculer=["answer_relevancy"]
    elif which_metrics=="contextRecall":
            metrics_a_calculer=["context_recall"]
    elif which_metrics=="factualCorrectness":
            metrics_a_calculer=["context_precision"] 

    """if overwrite:
        metrics_a_calculer = metrics[:]
    else:
        metrics_a_calculer = [
            m for m in metrics if _est_vide(df.at[row_idx, m])
        ]

    if not metrics_a_calculer:
        logger.info(
            "Toutes les métriques demandées sont déjà calculées pour "
            "cette question."
        )
        return _extraire_scores(df.loc[row_idx], metrics)

    logger.info(
        "Calcul des métriques manquantes : %s", which_metrics
    )
"""
    # --- 5. Construire le dataset RAGAS ---
    from ragas import EvaluationDataset, evaluate
    # IMPORTANT DE GARDER CES METADATA
    sample = {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": _normaliser_contextes(contexts),
        "reference": ground_truth,
    }
    dataset = EvaluationDataset.from_list([sample])

    # --- 6. Évaluer ---
    """metrics_instances = _construire_metriques(
        metrics_a_calculer,
        llm_judge,
        embeddings,
    )"""

    """from ragas.embeddings.base import embedding_factory
    embedder = embedding_factory(
                "BAAI/bge-m3",
                provider="huggingface"
            )"""
    
    from ragas.embeddings import HuggingFaceEmbeddings
    embedder = HuggingFaceEmbeddings(
    model="BAAI/bge-m3",
    use_api=False   # False = modèle local via sentence-transformers, True = API HuggingFace hébergée
)
    if which_metrics== "faith":
        from ragas.metrics import (
        Faithfulness
    )
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness()],
            llm=llm_judge,
            embeddings= embedder,
            batch_size=batch_size,
            raise_exceptions=False,
            show_progress=True,
        )
        
        result_df = result.to_pandas()
        valeur=result_df.iloc[0,-1]
    elif which_metrics== "answer":
            from ragas.metrics import (
           ResponseRelevancy
        )
            result = evaluate(
                dataset=dataset,
                metrics=[ResponseRelevancy()],
                llm=llm_judge,
                embeddings= embedder,
                batch_size=batch_size,
                raise_exceptions=False,
                show_progress=True,
            )

            result_df = result.to_pandas()
            valeur=result_df.iloc[0,-1]

    elif which_metrics== "contextRecall":
        from ragas.metrics import (
        LLMContextRecall
    )
        result = evaluate(
            dataset=dataset,
            metrics=[LLMContextRecall()],
            llm=llm_judge,
            embeddings= embedder,
            batch_size=batch_size,
            raise_exceptions=False,
            show_progress=True,
        )
        result_df = result.to_pandas()
        valeur=result_df.iloc[0,-1]
    
    elif which_metrics== "factualCorrectness":
            from ragas.metrics import (
            FactualCorrectness
        )
            result = evaluate(
                dataset=dataset,
                metrics=[FactualCorrectness()],
                llm=llm_judge,
                embeddings= embedder,
                batch_size=batch_size,
                raise_exceptions=False,
                show_progress=True,
            )
            result_df = result.to_pandas()
            #valeur=result_df["context_precision"].iloc[0]
            valeur=result_df.iloc[0,-1]
    print("le resultat",result, "et la valeur",valeur)

    #llm=llm_judge,
    #embeddings= embeddings,
    result_df = result.to_pandas()

    # --- 7. Mettre à jour les cellules ---
    for nom in metrics_a_calculer:
        print("le {nom} est bien dans metrics_a_calculer")
        
        df.at[row_idx, nom] = (
            round(float(valeur), 3)
            if not _est_vide(valeur)
            else pd.NA
        )
        print(f"Sauvegarde dans {nom} reussi")

    # --- 8. Recalculer score_moyen et flag ---
    metriques_presentes = [
        c for c in _METRIQUES_VALIDES if not _est_vide(df.at[row_idx, c])
    ]
    if metriques_presentes:
        moy = (
            df.loc[row_idx, metriques_presentes]
            .astype(float)
            .mean()
        )
        df.at[row_idx, "score_moyen"] = round(moy, 3)
        if moy >= 0.75:
            df.at[row_idx, "flag"] = "✅ OK"
        elif moy >= 0.5:
            df.at[row_idx, "flag"] = "⚠️ faible"
        else:
            df.at[row_idx, "flag"] = "❌ échec"

    # --- 9. Sauvegarder ---
    df.to_csv(csv_file, index=False, encoding="utf-8")
    logger.info("CSV sauvegardé : %s", csv_file)

    return _extraire_scores(df.loc[row_idx], metrics_a_calculer)

if __name__ == "__main__":
    from utils.eval_ragas import RagasEval 
    import json

    sample_queries= ["Informations sur les horaires d'ouverture de la mairie de Trifouillis-sur-Loire",
        "Dates cérémonies commémoratives officielles",
        "quel est le numéro de téléphone de la mairie ?",
        "Informations sur les démarches administratives pour déclarer une naissance",
        "quelles sont les démarches pour obtenir un permis de construire ?",
        "quels sont les horaires pour effectuer des travaux de bricolage ou de jardinage?",
        "quelles sont les règles concernant les animaux domestiques dans la commune ?",
        "Horaires pour vendre des boissons alcoolisées à emporter?",
        "Dans quel mois est célébré l'accueil des nouveaux arrivants dans la commune ?",
        "Donne moi le lieu les horaires et le contact de la police municipale de Trifouillis-sur-Loire",
        "Bonjour, quel temps fait-il aujourd'hui ?",
        "quel est le nom du maire de  Trifouillis-sur-Loire.?"
]

    expected_responses=["La mairie de Trifouillis-sur-Loire est ouverte au public du lundi au vendredi de 9h00 à 12h00 et de 14h00 à 17h00, ainsi que le samedi de 9h00 à 12h00.",
                    "19 mars : Journée nationale du souvenir (guerre d'Algérie), Dernier dimanche d'avril : Journée nationale du souvenir des victimes de la déportation,8 mai : Commémoration de la victoire de 1945, 27 mai : Journée nationale de la Résistance, 18 juin : Appel du Général de Gaulle, 14 juillet : Fête nationale, 11 novembre : Armistice 1918",
                    "La mairie ne possède pas de numéro de téléphone spécifique mais consulter le site internet de la commune www.trifouillis-sur-loire.fr",
                    "Les déclarations de naissance doivent être effectuées à la mairie dans les cinq jours qui suivent l'accouchement.",
                    "Pour obtenir un permis de construire, il est nécessaire de déposer une demande auprès du service urbanisme de la mairie, accompagnée des plans et documents requis.",
                    "Les travaux de bricolage ou de jardinage sont autorisés du lundi au vendredi de 8h00 à 20h00, le samedi de 9h00 à 19h00 et le dimanche de 10h00 à 18h00.",
                    "Les propriétaires d'animaux sont civilement responsables des dommages que leurs animaux pourraient causer à autrui, même lorsqu'ils sont échappés ou égarés. Il est interdit de laisser divaguer les chiens sur la voie publique, seuls et sans maître. Les chiens doivent être tenus en laisse dans les lieux publics. Les chiens de catégorie 1 et 2 (chiens d'attaque et chiens de garde ou de défense) doivent être déclarés en mairie et faire l'objet d'un permis de détention. Ils doivent être muselés et tenus en laisse par une personne majeure sur la voie publique. Il est interdit d'introduire des chiens, même tenus en laisse, dans les parcs et jardins publics, à l'exception des chiens guides d'aveugles et des chiens d'assistance aux personnes handicapées. Les animaux errants seront capturés et conduits à la fourrière animale intercommunale. Les frais de capture et de garde sont à la charge du propriétaire. La détention d'animaux non domestiques (NAC) est soumise à autorisation préalable et doit respecter la réglementation en vigueur."
                    "La vente à emporter de boissons alcoolisées est interdite entre 22h00 et 6h00",
                    "L'accueil des nouveaux habitants est organisé chaque année en septembre.",
                    "La police municipale de Trifouillis-sur-Loire est située au  Place de la République,. Les horaires d'ouverture sont du lundi au vendredi de 9h00 à 12h00 et de 14h00 à 17h00. La police municipale peut être contactée par téléphone au 02.XX.XX.XX.XX pendant les heures de service ou par mail à police@trifouillis-sur-loire.fr",
                    "Le système ne peut pas donner la météo en temps réel et suggère de consulter une source externe.",
                    "Son nom est Madame Pétillante Rigolade." 
    ]

    eval_rags=RagasEval()
    rag_dataset =eval_rags.create_data_rag(sample_queries,expected_responses)

    with open("ma_liste.json", "w", encoding="utf-8") as f:
        json.dump(rag_dataset, f, ensure_ascii=False, indent=2)

    with open("ma_liste.json", "r", encoding="utf-8") as f:
        ma_liste = json.load(f)

    for indx in range(len(sample_queries)):
        print(f"****************************Evaluation {indx}***************************")
        which_metrics="faith" #"faithAndAnswer"
        evaluer_et_sauvegarder(indx,eval_rags,dataset=ma_liste,which_metrics=which_metrics )

        which_metrics="contextRecall" 
        evaluer_et_sauvegarder(indx,eval_rags,dataset=ma_liste,which_metrics=which_metrics )

        which_metrics="factualCorrectness" 
        evaluer_et_sauvegarder(indx,eval_rags,dataset=ma_liste,which_metrics=which_metrics )

     