import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# ==========================================
# 1. LOAD ARTIFACTS
# ==========================================
@st.cache_resource # Caches the model so it doesn't reload on every button click
def load_artifacts():
    model = tf.keras.models.load_model('cnn_heart_model.keras')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    expected_cols = pickle.load(open('expected_columns.pkl', 'rb'))
    return model, scaler, expected_cols

model, scaler, expected_columns = load_artifacts()

# ==========================================
# 2. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Deep Learning Cardiology", page_icon="🫀", layout="centered")
st.title("🫀 1D-CNN Heart Disease Predictor")
st.markdown("Enter patient vitals below to run a real-time deep learning inference.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    submit = st.form_submit_button("Run Neural Network")

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
if submit:
    # 1. Capture Input
    input_dict = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
        'chol': chol, 'thalach': thalach, 'exang': exang, 
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    input_df = pd.DataFrame([input_dict])

    # 2. Scale Continuous Variables
    cont_val = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df[cont_val] = scaler.transform(input_df[cont_val])

    # 3. One-Hot Encode Categorical Variables
    cate_val = ['cp', 'exang', 'slope', 'ca', 'thal']
    input_df = pd.get_dummies(input_df, columns=cate_val, drop_first=True, dtype='float32')

    # 4. Align columns with training data (fills missing dummy columns with 0)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # 5. Reshape to 3D Tensor for CNN (batch_size, features, channels)
    input_tensor = np.asarray(input_df).astype('float32').reshape((1, len(expected_columns), 1))

    # 6. Inference
    prediction_prob = model.predict(input_tensor)[0][0]
    
    st.divider()
    if prediction_prob > 0.5:
        st.error(f"⚠️ **High Risk Detected:** The CNN predicts the presence of heart disease with {(prediction_prob*100):.1f}% confidence.")
    else:
        st.success(f"✅ **Low Risk:** The CNN predicts no heart disease (Confidence: {((1-prediction_prob)*100):.1f}%).")