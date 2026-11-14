from pathlib import Path
from io import BytesIO
from html import escape
import os
import sys

import altair as alt
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

ROOT = Path(os.environ.get("JURY_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluate import evaluate_model

ACCENT = "#93C5B5"
PANEL = "#141E27"
ORDER = ["Low", "Medium", "High"]


def panel(kicker, title, note):
    st.markdown(f'<div class="panel-kicker">{kicker}</div><div class="panel-title">{title}</div><div class="panel-note">{note}</div>', unsafe_allow_html=True)


def chart_theme(chart):
    return chart.configure(background=PANEL).configure_view(strokeWidth=0).configure_axis(
        labelColor="#B2C1CC", titleColor="#B2C1CC", gridColor="#26343F",
        domain=False, tickColor="#26343F", labelFontSize=11,
    ).configure_legend(labelColor="#B2C1CC", titleColor="#B2C1CC", orient="top")


st.set_page_config(page_title="Campus — Observatoire étudiant", page_icon="C", layout="wide")
st.markdown("<style>" + (Path(__file__).with_name("style.css")).read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
try:
    bundle = joblib.load(BytesIO((ROOT / "models/selected_model.joblib").read_bytes()))
except (OSError, ValueError) as error:
    st.error("Le modèle n’est pas disponible : " + str(error))
    st.stop()
model_name = bundle.get("model_name", "Logistic Regression")
model_label = "Régression logistique" if model_name == "Logistic Regression" else str(model_name)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">c</div><div class="brand-name">campus<small>Observatoire étudiant</small></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="side-label">Espace d’analyse</div><div class="side-value">Bien-être étudiant</div><div class="side-note">Une lecture des données et des performances du prototype.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="side-block"><div class="side-label">Modèle retenu</div><div class="side-value"><span class="dot"></span>{escape(model_label)}</div><div class="side-note">Sélection fondée sur le F1 de validation croisée.</div></div>', unsafe_allow_html=True)
    training_rows = int(bundle.get("training_rows", 1600))
    st.markdown(f'<div class="side-block"><div class="side-label">Repères du projet</div><div class="side-value">{training_rows:,} profils d’apprentissage<br>3 classes du dataset</div><div class="side-note">Low · Medium · High<br>Les classes ne sont pas des diagnostics.</div></div>'.replace(",", " "), unsafe_allow_html=True)
    st.caption("Jury blanc · Projet Data Science")

st.markdown('<div class="hero"><div><div class="eyebrow">Données étudiantes / Classification</div><h1 class="hero-title">Analyse du bien-être étudiant.</h1><p class="hero-description">Évaluer les résultats du modèle et explorer un profil fictif.</p></div><div class="hero-badge">Prototype pédagogique</div></div>', unsafe_allow_html=True)
st.markdown('<div class="notice"><strong>Cadre d’utilisation.</strong> Ce prototype ne constitue ni un diagnostic, ni un dépistage clinique. Aucune utilisation pour la sélection ou une décision disciplinaire.</div>', unsafe_allow_html=True)
performance, prediction = st.tabs(["Vue d’ensemble", "Simulation d’un profil"])

with performance:
    try:
        data = pd.read_csv(ROOT / "data/raw/student_mental_health.csv")
        data = data.drop_duplicates().dropna(subset=["Mental_Health_Risk"])
        X = data.drop(columns=["Student_ID", "Mental_Health_Risk"])
        y = data["Mental_Health_Risk"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
        labels = bundle["target_encoder"].transform(y_test.to_frame()).astype(int).ravel()
        result = evaluate_model(bundle["pipeline"], X_test, labels, bundle["class_names"])
        st.markdown(f'<div class="page-intro"><div><div class="page-title">Performance du modèle</div><div class="page-note">Résultats sur le test historique. Ce bilan ne représente pas une nouvelle évaluation indépendante.</div></div><div class="dataset-tag">{len(X_test)} profils · Test 20 %</div></div>', unsafe_allow_html=True)
        columns = st.columns(3, gap="medium")
        cards = [("Accuracy", "Exactitude", "Part des classes correctement prédites."),
                 ("F1 macro", "F1 macro", "Compromis précision / rappel, avec le même poids pour chaque classe."),
                 ("ROC AUC macro OvR", "ROC-AUC", "Classement des classes à partir des probabilités, sans garantie de calibration.")]
        for column, (key, label, help_text) in zip(columns, cards):
            column.metric(label, f"{result['metrics'][key]:.1%}", help=help_text)
        st.write("")
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            with st.container(border=True):
                panel("01 / Prédictions", "Classes observées et prédites", "Lignes : classe réelle. Colonnes : classe prédite.")
                matrix = result["confusion_matrix"].reindex(index=ORDER, columns=ORDER)
                cmap = LinearSegmentedColormap.from_list("campus", ["#15222C", "#34574F", ACCENT])
                fig, ax = plt.subplots(figsize=(5, 3.5), facecolor=PANEL)
                ax.set_facecolor(PANEL)
                ConfusionMatrixDisplay(matrix.to_numpy(), display_labels=ORDER).plot(ax=ax, cmap=cmap, colorbar=False, text_kw={"fontsize": 13, "fontweight": "normal"})
                ax.set_xlabel("Classe prédite", color="#A4B8C5", fontsize=10, labelpad=10)
                ax.set_ylabel("Classe réelle", color="#A4B8C5", fontsize=10, labelpad=8)
                ax.tick_params(colors="#CBD7DF", length=0, labelsize=10, pad=9)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.tight_layout(pad=1)
                st.pyplot(fig)
                plt.close(fig)
        with right:
            with st.container(border=True):
                panel("02 / Lecture par classe", "Équilibre des performances", "Un score global ne suffit pas : comparer les trois classes.")
                scores = result["report"].loc[ORDER, ["precision", "recall", "f1-score", "support"]].copy()
                chart_data = scores.reset_index(names="Classe").rename(columns={"f1-score": "F1"})
                chart = alt.Chart(chart_data).mark_bar(color=ACCENT, size=16, cornerRadiusEnd=3).encode(
                    x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                    y=alt.Y("Classe:N", sort=ORDER, title=None),
                    tooltip=["Classe:N", alt.Tooltip("F1:Q", format=".1%"), alt.Tooltip("support:Q", title="Profils", format=".0f")],
                ).properties(height=160)
                st.altair_chart(chart_theme(chart), width="stretch")
                table = scores.rename(columns={"precision": "Précision %", "recall": "Rappel %", "f1-score": "F1 %", "support": "Effectif"})
                table.iloc[:, :3] = (table.iloc[:, :3] * 100).round(1)
                table["Effectif"] = table["Effectif"].astype(int)
                st.dataframe(table, width="stretch")
    except (OSError, ValueError, KeyError) as error:
        st.info("Le bilan n’est pas disponible : " + str(error))

    with st.expander("Comparaison historique des modèles"):
        metrics_file = ROOT / "metrique/validation_test_metrics.csv"
        if metrics_file.exists():
            metrics = pd.read_csv(metrics_file)
            st.caption("Archive du notebook : ces chiffres décrivent les ajustements de cette expérience.")
            comparison_chart = alt.Chart(metrics).mark_bar(cornerRadiusEnd=2).encode(
                x=alt.X("Modele:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="Ensemble:N",
                y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                color=alt.Color("Ensemble:N", scale=alt.Scale(domain=["Train", "Validation CV", "Test"], range=["#627B8D", ACCENT, "#C6AE87"]), title=None),
                tooltip=["Modele:N", "Ensemble:N", alt.Tooltip("Accuracy:Q", format=".1%")],
            ).properties(height=230)
            st.altair_chart(chart_theme(comparison_chart), width="stretch")
            st.dataframe(metrics.round(4), width="stretch", hide_index=True)
            st.download_button("Télécharger les métriques CSV", metrics.to_csv(index=False).encode("utf-8"), "metriques.csv", "text/csv")

with prediction:
    st.markdown('<div class="page-intro"><div><div class="page-title">Explorer un profil fictif</div><div class="page-note">Renseigner les variables du dataset, puis observer la classe prédite.</div></div><div class="dataset-tag">15 variables</div></div>', unsafe_allow_html=True)
    numeric_fields = {
        "Age": ("Âge (ans)", 22), "Year_of_Study": ("Année universitaire", 3),
        "Sleep_Hours": ("Sommeil (heures)", 7.0), "Study_Hours_Per_Week": ("Études (heures / semaine)", 20.0),
        "GPA": ("GPA", 2.8), "Social_Media_Hours": ("Réseaux sociaux (heures)", 4.0),
        "Screen_Time_Hours": ("Écran (heures)", 7.0), "Financial_Stress": ("Stress financier", 5.0),
        "Family_Support": ("Soutien familial", 7.0), "Attendance_Rate": ("Présence (%)", 85.0),
    }
    category_labels = {"Gender": "Genre", "Physical_Activity": "Activité physique", "Part_Time_Job": "Emploi à temps partiel", "Diet_Quality": "Qualité de l’alimentation", "Previous_Mental_Health_History": "Antécédents déclarés"}
    translations = {"Female": "Femme", "Male": "Homme", "Non-binary": "Non binaire", "No": "Non", "Yes": "Oui", "Low": "Faible", "Moderate": "Modérée", "High": "Élevée", "Poor": "Faible", "Average": "Moyenne", "Good": "Bonne"}
    prep = bundle["pipeline"].named_steps["preprocessing"].named_steps["preparation"]
    category_pipeline = prep.named_transformers_["categorie"]
    names = next(columns for name, _, columns in prep.transformers_ if name == "categorie")
    categories = dict(zip(names, category_pipeline.named_steps["encodage"].categories_))
    inputs = {}

    def input_field(name):
        if name in numeric_fields:
            label, default = numeric_fields[name]
            if isinstance(default, int):
                return st.number_input(label, min_value=1, value=default, step=1, key=name)
            maximum = 100.0 if name == "Attendance_Rate" else 168.0 if name == "Study_Hours_Per_Week" else None
            return st.number_input(label, min_value=0.0, max_value=maximum, value=default, step=0.1, key=name)
        choices = [str(value) for value in categories[name] if not pd.isna(value)]
        return st.selectbox(category_labels[name], choices, format_func=lambda value: translations.get(value, value), key=name)

    groups = [
        ("01", "Profil & parcours", ["Age", "Year_of_Study", "Gender", "GPA", "Attendance_Rate", "Part_Time_Job"]),
        ("02", "Habitudes quotidiennes", ["Sleep_Hours", "Study_Hours_Per_Week", "Social_Media_Hours", "Screen_Time_Hours", "Physical_Activity", "Diet_Quality"]),
        ("03", "Contexte & soutien", ["Financial_Stress", "Family_Support", "Previous_Mental_Health_History"]),
    ]
    with st.form("prediction"):
        for column, (number, title, fields) in zip(st.columns(3, gap="medium"), groups):
            with column:
                with st.container(border=True):
                    st.markdown(f'<div class="form-heading"><span>{number}</span>{title}</div>', unsafe_allow_html=True)
                    for name in fields:
                        inputs[name] = input_field(name)
        st.caption("Sommeil, écran et réseaux : hypothèse h/jour à confirmer. GPA et stress/soutien : conserver les unités du dataset.")
        submitted = st.form_submit_button("Analyser le profil")
    if submitted:
        try:
            frame = pd.DataFrame([inputs])
            if not np.isfinite(frame.select_dtypes(include="number").to_numpy()).all():
                raise ValueError("Les nombres doivent être finis")
            codes = bundle["pipeline"].predict(frame)
            label = bundle["target_encoder"].inverse_transform(codes.astype(int).reshape(-1, 1)).ravel()[0]
            probabilities = bundle["pipeline"].predict_proba(frame)[0]
            st.markdown('<div class="result-heading">Résultat de la simulation</div>', unsafe_allow_html=True)
            st.success("Classe prédite du dataset : " + str(label))
            values = pd.DataFrame({"Classe": bundle["class_names"], "Probabilité": probabilities})
            chart = alt.Chart(values).mark_bar(color=ACCENT, size=18, cornerRadiusEnd=3).encode(
                x=alt.X("Probabilité:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                y=alt.Y("Classe:N", sort=ORDER, title=None),
                tooltip=["Classe:N", alt.Tooltip("Probabilité:Q", format=".1%")],
            ).properties(height=160)
            st.altair_chart(chart_theme(chart), width="stretch")
            st.caption("Une sortie du prototype, pas une probabilité clinique.")
        except ValueError as error:
            st.error("Vérifier la saisie : " + str(error))

st.markdown('<div class="footer">CAMPUS / OBSERVATOIRE ÉTUDIANT &nbsp;·&nbsp; Projet Data Science — Jury blanc</div>', unsafe_allow_html=True)
