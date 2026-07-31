import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# CONFIGURATION GÉNÉRALE
# ============================================================

#Configurer l'icône et le titre de l'onglet du navigateur
icon = Image.open("logo_DLR.PNG")

st.set_page_config(
    page_title="🌍 DJE-Gueu | Plateforme numérique",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE VISUEL
# ============================================================

st.markdown("""
<style>

    /* Arrière-plan général */
    .main {
        background-color: #f7f9fc;
    }

    /* Titre principal */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #12355B;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 20px;
        color: #4F5D75;
        margin-bottom: 30px;
    }

    /* Cartes */
    .card {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* KPI */
    .kpi {
        padding: 20px;
        border-radius: 12px;
        background-color: white;
        text-align: center;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    }

    .kpi-title {
        font-size: 15px;
        color: #6c757d;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        color: #12355B;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #102A43;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODÈLE DJE-GUEU
# ============================================================

def dje_gueu_model(Y, lambd, alpha, kappa, mu):
    """
    Équation différentielle du modèle DJE-Gueu :

    dY/dt = λY + αY(1 - exp(-κY)) - μY²
    """

    return (
        lambd * Y
        + alpha * Y * (1 - np.exp(-kappa * Y))
        - mu * Y**2
    )


# ============================================================
# SIMULATION NUMÉRIQUE
# ============================================================

def simulate_model(
    Y0,
    lambd,
    alpha,
    kappa,
    mu,
    T=50,
    dt=0.05
):

    time = np.arange(0, T + dt, dt)

    Y = np.zeros(len(time))
    Y[0] = Y0

    for i in range(1, len(time)):

        derivative = dje_gueu_model(
            Y[i-1],
            lambd,
            alpha,
            kappa,
            mu
        )

        Y[i] = Y[i-1] + dt * derivative

        # Éviter les valeurs négatives
        if Y[i] < 0:
            Y[i] = 0

    return time, Y


# ============================================================
# ANALYSE DES ÉQUILIBRES
# ============================================================

def find_equilibria(
    lambd,
    alpha,
    kappa,
    mu,
    Ymax=100,
    points=10000
):

    Y_values = np.linspace(0, Ymax, points)

    F = dje_gueu_model(
        Y_values,
        lambd,
        alpha,
        kappa,
        mu
    )

    equilibria = []

    for i in range(len(Y_values)-1):

        if F[i] == 0:
            equilibria.append(Y_values[i])

        elif F[i] * F[i+1] < 0:

            root = (
                Y_values[i]
                -
                F[i]
                *
                (Y_values[i+1] - Y_values[i])
                /
                (F[i+1] - F[i])
            )

            equilibria.append(root)

    return np.unique(
        np.round(equilibria, 4)
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🌍 DJE-Gueu"
    )

    st.caption(
        "Plateforme numérique de modélisation "
        "de la dynamique de jeunesse et d'emploi"
    )

    st.markdown("---")

    menu = st.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "📖 Le modèle DJE-Gueu",
            "🧮 Simulateur",
            "📊 Équilibres",
            "📈 Stabilité",
            "🔬 Sensibilité",
            "🌀 Dynamique",
            "🌍 Scénarios territoriaux",
            "🎯 Politiques publiques",
            "📋 Indicateurs décideurs",
            "🌐 ODD & Agenda 2030",
            "📚 Documentation",
            "ℹ️ À propos"
        ]
    )

    st.markdown("---")

    st.caption(
        "DJE-Gueu Model\n\n"
        "Concepteur : Guy Ghislain GUEU\n\n"
        "Création : 23 février 2026"
    )


# ============================================================
# PAGE ACCUEIL
# ============================================================

if menu == "🏠 Accueil":

    st.markdown(
        '<div class="main-title">'
        'Modèle DJE-Gueu'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Plateforme numérique d’analyse de la dynamique '
        'de jeunesse et d’emploi en Afrique'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Cette plateforme transforme le modèle mathématique "
        "DJE-Gueu en un instrument numérique d'exploration, "
        "de simulation et d'aide à l'analyse."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Modèle",
            "DJE-Gueu"
        )

    with col2:
        st.metric(
            "Type",
            "Non linéaire"
        )

    with col3:
        st.metric(
            "Variable",
            "Y(t)"
        )

    with col4:
        st.metric(
            "Paramètres",
            "λ α κ μ"
        )

    st.markdown("---")

    st.markdown("### 🎯 Objectif de la plateforme")

    st.write(
        """
        Le modèle DJE-Gueu propose un cadre mathématique
        permettant d'explorer la dynamique de la population
        jeune économiquement active et d'étudier l'influence
        des politiques et initiatives favorisant l'insertion
        professionnelle.
        """
    )

    st.markdown("### 🌍 Publics concernés")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            "#### 🏛️ Décideurs publics\n"
            "Analyse de scénarios et politiques d'emploi."
        )

    with col2:
        st.markdown(
            "#### 🌐 Institutions internationales\n"
            "Exploration des dynamiques territoriales et "
            "des objectifs de développement."
        )

    with col3:
        st.markdown(
            "#### 🎓 Chercheurs\n"
            "Simulation, analyse mathématique et validation."
        )


# ============================================================
# PAGE MODÈLE
# ============================================================

elif menu == "📖 Le modèle DJE-Gueu":

    st.title(
        "📖 Présentation du modèle DJE-Gueu"
    )

    st.markdown(
        """
        Le modèle DJE-Gueu — **Modèle Gueu de Dynamique
        de Jeunesse et d'Emploi** — est un modèle mathématique
        dynamique fondé sur une équation différentielle
        non linéaire du premier ordre.
        """
    )

    st.latex(
        r"""
        \frac{dY}{dt}
        =
        \lambda Y
        +
        \alpha Y(1-e^{-\kappa Y})
        -
        \mu Y^2
        """
    )

    st.markdown("### Variables et paramètres")

    data = pd.DataFrame({
        "Symbole": [
            "Y(t)",
            "λ",
            "α",
            "κ",
            "μ"
        ],
        "Signification": [
            "Population jeune économiquement active",
            "Taux de croissance / entrée",
            "Intensité des politiques et initiatives",
            "Vitesse d'activation des opportunités",
            "Effet de saturation"
        ]
    })

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE SIMULATEUR
# ============================================================

elif menu == "🧮 Simulateur":

    st.title(
        "🧮 Simulateur DJE-Gueu"
    )

    st.write(
        "Modifiez les paramètres du modèle et observez "
        "la trajectoire dynamique de Y(t)."
    )

    col1, col2 = st.columns(2)

    with col1:

        Y0 = st.slider(
            "Population initiale Y₀",
            0.1,
            50.0,
            5.0
        )

        lambd = st.slider(
            "λ — croissance",
            0.0,
            0.20,
            0.03,
            step=0.005
        )

        alpha = st.slider(
            "α — politiques / initiatives",
            0.0,
            0.20,
            0.02,
            step=0.005
        )

    with col2:

        kappa = st.slider(
            "κ — activation",
            0.01,
            2.0,
            0.5,
            step=0.01
        )

        mu = st.slider(
            "μ — saturation",
            0.001,
            0.20,
            0.04,
            step=0.005
        )

        T = st.slider(
            "Horizon temporel",
            10,
            200,
            50
        )

    time, Y = simulate_model(
        Y0,
        lambd,
        alpha,
        kappa,
        mu,
        T
    )

    fig, ax = plt.subplots()

    ax.plot(
        time,
        Y
    )

    ax.set_title(
        "Trajectoire dynamique du modèle DJE-Gueu"
    )

    ax.set_xlabel(
        "Temps"
    )

    ax.set_ylabel(
        "Y(t)"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    st.pyplot(fig)

    st.success(
        f"Valeur finale estimée : {Y[-1]:.4f}"
    )


# ============================================================
# PAGE ÉQUILIBRES
# ============================================================

elif menu == "📊 Équilibres":

    st.title(
        "📊 Analyse des équilibres"
    )

    st.write(
        """
        Cette section recherche les valeurs de Y pour lesquelles
        la dynamique du système devient stationnaire.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        lambd = st.number_input(
            "λ",
            value=0.03
        )

    with col2:
        alpha = st.number_input(
            "α",
            value=0.02
        )

    with col3:
        kappa = st.number_input(
            "κ",
            value=0.5
        )

    with col4:
        mu = st.number_input(
            "μ",
            value=0.04
        )

    equilibria = find_equilibria(
        lambd,
        alpha,
        kappa,
        mu
    )

    st.markdown(
        "### Équilibres détectés"
    )

    if len(equilibria) > 0:

        for eq in equilibria:

            st.metric(
                "Équilibre",
                f"Y* = {eq}"
            )

    else:

        st.warning(
            "Aucun équilibre détecté dans la plage étudiée."
        )


# ============================================================
# PAGE STABILITÉ
# ============================================================

elif menu == "📈 Stabilité":

    st.title(
        "📈 Analyse de stabilité"
    )

    st.info(
        """
        Cette section peut être développée pour calculer
        la dérivée de la fonction dynamique autour de chaque
        équilibre et déterminer sa stabilité locale.
        """
    )

    st.markdown(
        """
        Pour un modèle scalaire :

        - \(F'(Y^*) < 0\) : équilibre localement stable.
        - \(F'(Y^*) > 0\) : équilibre localement instable.
        """
    )


# ============================================================
# PAGE SENSIBILITÉ
# ============================================================

elif menu == "🔬 Sensibilité":

    st.title(
        "🔬 Analyse de sensibilité"
    )

    st.write(
        """
        Cette section permet d'étudier comment les résultats
        du modèle évoluent lorsque les paramètres changent.
        """
    )

    st.info(
        "Extension recommandée : graphiques automatiques "
        "pour α, κ et μ."
    )


# ============================================================
# PAGE DYNAMIQUE
# ============================================================

elif menu == "🌀 Dynamique":

    st.title(
        "🌀 Dynamique du système"
    )

    st.write(
        """
        Cette section pourra intégrer :

        • trajectoires temporelles ;

        • champ de direction ;

        • diagrammes de phase ;

        • isoclines ;

        • portraits dynamiques.
        """
    )


# ============================================================
# PAGE SCÉNARIOS TERRITORIAUX
# ============================================================

elif menu == "🌍 Scénarios territoriaux":

    st.title(
        "🌍 Scénarios territoriaux"
    )

    territoire = st.selectbox(
        "Sélectionner un territoire",
        [
            "Côte d'Ivoire",
            "Guinée",
            "Sénégal",
            "Togo",
            "Autre territoire"
        ]
    )

    st.write(
        f"Analyse actuellement configurée pour : "
        f"**{territoire}**"
    )

    st.info(
        "Cette section pourra être connectée ultérieurement "
        "à des données statistiques réelles."
    )


# ============================================================
# PAGE POLITIQUES PUBLIQUES
# ============================================================

elif menu == "🎯 Politiques publiques":

    st.title(
        "🎯 Simulateur de politiques publiques"
    )

    st.write(
        """
        L'objectif est de comparer différents scénarios
        d'intervention publique.
        """
    )

    scenario = st.selectbox(
        "Scénario",
        [
            "Situation de référence",
            "Politique faible",
            "Politique intermédiaire",
            "Politique renforcée"
        ]
    )

    st.success(
        f"Scénario sélectionné : {scenario}"
    )


# ============================================================
# PAGE INDICATEURS
# ============================================================

elif menu == "📋 Indicateurs décideurs":

    st.title(
        "📋 Tableau de bord pour décideurs"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dynamique",
            "Positive"
        )

    with col2:
        st.metric(
            "Équilibre",
            "À analyser"
        )

    with col3:
        st.metric(
            "Risque de saturation",
            "Modéré"
        )

    st.warning(
        "Les indicateurs présentés ici sont des démonstrations "
        "et ne doivent pas être interprétés comme des résultats "
        "empiriques sans calibration sur des données réelles."
    )


# ============================================================
# PAGE ODD
# ============================================================

elif menu == "🌐 ODD & Agenda 2030":

    st.title(
        "🌐 DJE-Gueu et Objectifs de développement durable"
    )

    st.markdown(
        """
        Le modèle peut être étudié en relation avec plusieurs
        Objectifs de développement durable, notamment :

        **ODD 4** — Éducation de qualité

        **ODD 8** — Travail décent et croissance économique

        **ODD 10** — Réduction des inégalités

        **ODD 11** — Villes et communautés durables

        **ODD 17** — Partenariats pour la réalisation des objectifs
        """
    )


# ============================================================
# DOCUMENTATION
# ============================================================

elif menu == "📚 Documentation":

    st.title(
        "📚 Documentation scientifique"
    )

    st.markdown(
        """
        ### Ressources

        - Présentation du modèle
        - Hypothèses mathématiques
        - Méthodologie
        - Calibration
        - Validation
        - Simulation numérique
        - Analyse de stabilité
        - Analyse de sensibilité
        - Applications territoriales
        """
    )


# ============================================================
# À PROPOS
# ============================================================

elif menu == "ℹ️ À propos":

    st.title(
        "ℹ️ À propos du projet"
    )

    st.image("GUEU_labe1.jpg", use_container_width=True)

    st.markdown(
        """
        ## Modèle DJE-Gueu

        **Modèle Gueu de Dynamique de Jeunesse et d'Emploi**

        Concepteur : **Guy Ghislain GUEU**

        Date de création : **23 février 2026**

        Domaine : **Équations différentielles et modélisation
        mathématique appliquée**
        """
    )


st.html("<div style='height: 30px;'></div>") # Crée un espace vide de 30 pixels

st.markdown(
    """
    🔎 **Informations**

    Les travaux de recherche scientifique ont été réalisés par Guy Ghislain GUEU, 
    Scientifique indépendant en équations différentielles. 
    Ainsi que ses élèves-chercheurs du programme PESURS représentés à travers l'Afrique.

    🔎 **Contacts**
    
    (00 221) 77 807 62 07 uniquement par WhatsApp
    ghislainci@outlook.fr
    """
)
    st.info(
        "Le modèle doit être calibré et validé sur des données "
        "empiriques avant toute utilisation opérationnelle "
        "dans la prise de décision publique."
    )

st.html("<div style='height: 30px;'></div>") # Crée un espace vide de 30 pixels


st.markdown(
    """
    ** **
    """
)

st.markdown("Copyright © 2026 DiffLink Research.")
st.markdown("Tous droits réservés.")
