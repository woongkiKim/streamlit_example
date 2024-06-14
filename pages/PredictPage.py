import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import joblib



# 기본 데이터 프레임 생성
default_values = {
    'sepal_length': 0,
    'sepal_width': 0,
    'petal_length': 0,
    'petal_width': 0,
}

df = pd.DataFrame(default_values, index=[0])

# Streamlit 레이아웃
st.title("붓꽃 모델 예측 대시보드")

# 모델 불러오기
model = joblib.load('./iris_model.pkl')

# 첫 번째 구역: 개별 데이터 입력 및 결과 출력
st.header("개별 예측")

with st.form("🤔 내가 원하는 데이터 입력"):
    edited_df = st.data_editor(df, num_rows="dynamic")
    submit_button = st.form_submit_button(label="예측")

if submit_button:
    result_proba = model.predict_proba(edited_df)
    result = model.predict(edited_df)
    species_mapping = {
        0: 'Setosa',
        1: 'Versicolor',
        2: 'Virginica'
    }
    predicted_species = species_mapping[result[0]]

    st.write(f"예측 결과: {predicted_species}")

    st.write("각 품종에 속할 확률:")
    st.write(f"🌸 Setosa: {result_proba[0][0] * 100:.2f}%")
    st.write(f"🌼 Versicolor: {result_proba[0][1] * 100:.2f}%")
    st.write(f"🦀 Virginica: {result_proba[0][2] * 100:.2f}%")

