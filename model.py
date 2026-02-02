import streamlit as st
import numpy as np
import joblib
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import os

# ================= CONFIG =================
st.set_page_config(
    page_title="Student Depression Predictor",
    page_icon="🎓",
    layout="wide"
)

# ================= LOAD MODEL & SCALER =================
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")

# ================= LOAD OR CREATE ENCODER =================
encoder_path = "encoder.pkl"
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)
else:
    # Créer un encodeur pour Gender et Department si le fichier n'existe pas
    encoder = {}
    gender_le = LabelEncoder()
    gender_le.fit(["Male", "Female"])
    encoder["Gender"] = gender_le

    dept_le = LabelEncoder()
    dept_le.fit(["Science","Engineering","Business","Arts","Medical"])
    encoder["Department"] = dept_le

    joblib.dump(encoder, encoder_path)

# ================= CSS =================
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.card { background-color: white; padding: 25px; border-radius: 15px; box-shadow: 0px 4px 15px rgba(0,0,0,0.08); }
.result-ok { background-color: #eafaf1; padding: 15px; border-radius: 10px; color: #1e8449; font-weight: bold; }
.result-risk { background-color: #fdecea; padding: 15px; border-radius: 10px; color: #c0392b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("📌 À propos")
st.sidebar.write("""
Cette application prédit le **risque de dépression chez les étudiants**
à l’aide d’un **Kernel SVM (RBF)** optimisé.

📊 Données : Student Lifestyle  
☁️ Déploiement : Streamlit Cloud  
""")

image = Image.open("image/student.jpg")
st.sidebar.image(image, use_column_width=True)

st.sidebar.markdown("---")
st.sidebar.write("👩‍🎓 Projet Machine Learning")

# ================= MAIN =================
st.markdown("<h1 style='text-align:center;'>🎓 Student Depression Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Analyse basée sur le mode de vie étudiant</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧑 Informations personnelles")
    age = st.slider("Âge", 16, 40, 22)
    gender = st.selectbox("Sexe", ["Male", "Female"])
    department = st.selectbox("Département", ["Science", "Arts", "Commerce", "Engineering", "Medical"])
    cgpa = st.slider("CGPA (moyenne académique)", 0.0, 4.0, 3.0)

with col2:
    st.subheader("📚 Mode de vie")
    sleep = st.slider("Heures de sommeil / jour", 0.0, 12.0, 7.0)
    study = st.slider("Heures d’étude / jour", 0.0, 15.0, 5.0)
    social = st.slider("Temps sur réseaux sociaux (h)", 0.0, 12.0, 3.0)
    activity = st.slider("Activité physique (0 = faible, 10 = élevée)", 0, 10, 5)
    stress = st.slider("Niveau de stress (0 = faible, 10 = élevé)", 0, 10, 5)

st.markdown("---")

# ================= PREDICTION =================
if st.button("🔍 Lancer la prédiction"):

    # Encoder les valeurs utilisateur
    gender_enc = encoder["Gender"].transform([gender])[0]
    dept_enc = encoder["Department"].transform([department])[0]
    student_id = 0

    # Créer le vecteur de features et scaler
    X = np.array([[student_id, age, gender_enc, dept_enc, cgpa, sleep, study, social, activity, stress]])
    X_scaled = scaler.transform(X)

    # Prédiction et probabilité
    pred = model.predict(X_scaled)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0][1]
    else:
        proba = model.decision_function(X_scaled)[0]

    # Affichage
    if pred == 1:
        st.markdown(
            f"<div class='result-risk'>⚠️ Risque de dépression détecté<br>Score : {proba:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-ok'>✅ Aucun signe de dépression détecté<br>Score : {proba:.2f}</div>",
            unsafe_allow_html=True
        )

# ================= FOOTER =================
st.markdown("""
<hr>
<p style="text-align:center; color:gray;">
Projet académique – Machine Learning & Cloud Deployment
</p>
""", unsafe_allow_html=True)
