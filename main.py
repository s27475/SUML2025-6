import os
#problem macbookowy zbyt wiele watkow, liniki pod spodem to ograniczaja i rozwiazuja problem
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os

st.set_page_config(page_title="Aplikacja Wilks", layout="wide")

@st.cache_resource
def load_ag_model():
    model_path = "AutogluonModels/ag-20260118_231630"
    if os.path.exists(model_path):
        return TabularPredictor.load(model_path)
    else:
        return None

st.sidebar.title("Menu")
selected_option = st.sidebar.selectbox("Wybierz opcję", ["Strona główna", "Ranking zawodników", "Kalkulatory"])

if selected_option == "Strona główna":
    st.title("Projekt: Przewidywanie wyników WILKS")
    st.markdown(
        """
        ### Autorzy:
        Dominik Piwowarczyk (s27475), Jakub Buczek (s27348), Maciej Karolczak (s27606)

        ### Opis projektu
        Aplikacja umożliwia analizę wyników sportowców z bazy OpenPowerlifting w kontekście współczynnika Wilksa, który pozwala porównywać siłę zawodników niezależnie od ich masy ciała.

        ---

        ### Czym jest współczynnik Wilksa?
        Współczynnik Wilksa to matematyczny przelicznik opracowany w celu obiektywnego porównywania siły zawodników o różnej masie ciała.
        """
    )
    st.latex(r"""
    \text{Wilks Score} = \frac{500 \times \text{Total}}{a + bW + cW^2 + dW^3 + eW^4 + fW^5}
    """)
    st.markdown("### Tabela stałych współczynników zależnych od płci")
    coeffs = [
        {"Stała": "a", "Mężczyźni": -216.0475144, "Kobiety": 594.31747775582},
        {"Stała": "b", "Mężczyźni": 16.2606339, "Kobiety": -27.23842536447},
        {"Stała": "c", "Mężczyźni": -0.002388645, "Kobiety": 0.82112226871},
        {"Stała": "d", "Mężczyźni": -0.00113732, "Kobiety": -0.00930733913},
        {"Stała": "e", "Mężczyźni": 7.01863e-06, "Kobiety": 4.731582e-05},
        {"Stała": "f", "Mężczyźni": -1.291e-08, "Kobiety": -9.054e-08},
    ]
    st.table(coeffs)

elif selected_option == "Ranking zawodników":
    tab1, tab2 = st.tabs(["Mężczyźni", "Kobiety"])
    zawodnicy = pd.DataFrame([
        ["Adam", "Nowak", "M", 83.5, 420.4],
        ["Michał", "Kowalski", "M", 92.0, 398.7],
        ["Paweł", "Wiśniewski", "M", 74.8, 435.2],
        ["Marcin", "Dąbrowski", "M", 72.2, 438.1],
        ["Anna", "Kamińska", "K", 58.4, 372.5],
        ["Katarzyna", "Zawadzka", "K", 63.2, 388.7],
        ["Magdalena", "Pawlak", "K", 52.1, 400.3],
    ], columns=["Imię", "Nazwisko", "Płeć", "Waga (kg)", "Wynik Wilks"])

    mezczyzni = zawodnicy[zawodnicy["Płeć"] == "M"]
    kobiety = zawodnicy[zawodnicy["Płeć"] == "K"]

    with tab1:
        st.title("🏋️‍♂️ Ranking mężczyzn")
        st.dataframe(mezczyzni.sort_values(by="Wynik Wilks", ascending=False), use_container_width=True)
    with tab2:
        st.title("🏋️‍♀️ Ranking kobiet")
        st.dataframe(kobiety.sort_values(by="Wynik Wilks", ascending=False), use_container_width=True)

else:
    st.title("Kalkulatory")
    tab1, tab2 = st.tabs(['Kalkulator BMI', 'Kalkulator WILKS (Model AI)'])

    with tab1:
        st.write('### Kalkulator BMI')
        wzrost = st.slider('Wzrost (cm)', 100, 220, 175)
        waga_bmi = st.slider('Waga (kg)', 30, 200, 80)
        if st.button("Oblicz BMI"):
            bmi = waga_bmi / ((wzrost / 100) ** 2)
            st.info(f"Twoje BMI wynosi: {bmi:.2f}")

    with tab2:
        st.write("### Oblicz WILKS na podstawie modelu Machine Learning")
        st.write("Wprowadź dane, aby model przewidział Twój wynik.")

        plec = st.radio('Płeć', ['Mężczyzna', 'Kobieta'])
        waga = st.number_input('Masa ciała (kg)', min_value=30.0, value=85.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            przysiad = st.number_input("Przysiad (kg)", min_value=0.0, value=140.0)
        with col2:
            lawka = st.number_input("Wyciskanie (kg)", min_value=0.0, value=100.0)
        with col3:
            martwy = st.number_input("Martwy ciąg (kg)", min_value=0.0, value=180.0)

        total = przysiad + lawka + martwy
        st.metric("Twój łączny wynik (Total)", f"{total} kg")

        if st.button("Uruchom predykcję modelu"):
            predictor = load_ag_model()

            if predictor:
                input_df = pd.DataFrame({
                    'Sex': [1 if plec == 'Mężczyzna' else 0],
                    'BodyweightKg': [waga],
                    'BestSquatKg': [przysiad],
                    'BestBenchKg': [lawka],
                    'BestDeadliftKg': [martwy],
                    'TotalKg': [total]
                })

                prediction = predictor.predict(input_df)
                st.success(f"### Przewidywany Wilks: {prediction.iloc[0]:.2f}")
                st.balloons()
            else:
                st.error(
                    "Nie znaleziono zapisanego modelu w folderze 'AutogluonModels'. Uruchom najpierw notebook 'suml.ipynb'.")