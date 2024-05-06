
# Import modul
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import streamlit as st
import os
import time

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.plot import plot_missing_value
from jcopml.utils import save_model, load_model

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from gensim.models import FastText

# Norm Vec Function
def norm_sent_vector(sentence, ft_model):
    vecs = [ft_model[word.lower()] for word in word_tokenize(sentence)]
    norm_vecs = [vec / np.linalg.norm(vec) for vec in vecs if np.linalg.norm(vec) > 0]
    sent_vec = np.mean(norm_vecs, axis=0)
    return sent_vec

st.set_page_config(page_title="Analysis App",page_icon=":part_alternation_mark:", layout="wide")

# ----- Header and Data Input -----
with st.container():
    st.subheader("Hi, i am qal :wave:")
    st.title("Sentimen Analisis FastText")
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
        #folder = "D:\Code\Python\machine_learning\streamlit\data"
        #path = Path(folder, uploaded_file.name)
        with open(f"{uploaded_file.name}", mode='wb') as w:
            w.write(uploaded_file.getvalue())

# ----- Preview data -----   
with st.container():
    st.write("---")
    st.subheader("Preview data")
    
    # 1. Preprocessing
    df = pd.read_csv(f"{uploaded_file.name}")
    df.text = df.text.str.replace(r'[^0-9a-zA-Z\s]'," ", regex=True)
    df.text = df.text.str.replace(r'\s+'," ", regex=True)
    df.drop(df[df.text == ""].index, inplace=True)
    df.drop(df[df.text == " "].index, inplace=True)
    df.dropna(inplace=True)

    st.write(":green[Cleaned Data ....] :white_check_mark:")
    st.dataframe(df, use_container_width=True)
    st.write("data size", df.shape)

# 1.1 Binning rating
df.rating = pd.cut(df.rating, bins=[0,2,5], labels=["negatif","positif"])

# ----- Language Modeling ----- 
with st.container():
    st.write("---")
    st.subheader("Language Modeling")
    # 2. Language Modeling
    # 2.1 Load Model
    # ft_model = FastText.load("ft_model/review_product.fasttext")
    sentences = [word_tokenize(text.lower()) for text in df.text]
    ft_model = FastText(sentences, vector_size=128, window=5, min_count=3, workers=4, epochs=100, sg=0, hs=0)
    ft = ft_model.wv
    st.write(":green[Load Fastext ...] :white_check_mark:")

    left_col, right_col = st.columns(2)
    with left_col:
        text = st.text_input("Text Similarity", placeholder="input word to check")
        st.write(pd.DataFrame(ft.similar_by_word(text), columns=["word","similarity"]))
    
    with right_col:
        text_2 = st.text_input("Word to Vec", placeholder="input word to vec")
        word_vec = ft[text_2].tolist()
        # style = f'<p style="font-size: 12px;">{word_vec}</p>'
        # st.markdown(style, unsafe_allow_html=True)
        st.text(word_vec)

# ----- XGBoost Model Evaluation ----- 
with st.container():
    st.write("---")
    st.subheader("XGBoost Model & Evaluation")
    xgb_model = load_model('model/shop_recommender_v2.pkl')

    # 3. XGBoost Model
    # 3.1 Train Test Split
    vecs = [norm_sent_vector(sentence, ft) for sentence in df.text]
    vecs = np.array(vecs)

    X = vecs
    y = df.rating

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)

    st.write(":green[Load XGBoost Model ...] :white_check_mark:")

    st.write("X_train shape : ", X_train.shape)
    st.write("X_test shape : " , X_test.shape)
    st.write("y_train shape : ", y_train.shape)
    st.write("y_test shape : ", y_test.shape)
    st.write("Train Score : ", xgb_model.score(X_train, y_train))
    st.write("Test Score : ", xgb_model.score(X_test, y_test))
    st.write("Best Val Score : ", xgb_model.best_score_)
    st.write("Best Parameter : ", xgb_model.best_params_)
    
# ----- Shop Recommendation -----
with st.container():
    st.write("---")
    st.subheader("Shop Recommendation")

    # Split Data
    df_shop = dict(tuple(df.groupby(["shop_id", "category"])))
    df_shop = {k:v for k,v in df_shop.items() if (df_shop[k].shape[0] >= 10) & (df_shop[k].rating.value_counts()["negatif"] >= 3)}

    # encode label
    for k,v in df_shop.items():
        df_shop[k].rating = LabelEncoder().fit_transform(df_shop[k].rating)

    rank_shop = pd.DataFrame({
        "shop_id":[str(k[0]) for k in df_shop.keys()],
        "category":[k[1] for k in df_shop.keys()],
        "accuracy_score": [accuracy_score(df_shop[k].rating, xgb_model.predict(np.array([norm_sent_vector(sentence,ft) for sentence in df_shop[k].text]))) for k in df_shop.keys()],
        "f1_score": [f1_score(df_shop[k].rating, xgb_model.predict(np.array([norm_sent_vector(sentence,ft) for sentence in df_shop[k].text]))) for k in df_shop.keys()],
        "recall": [recall_score(df_shop[k].rating, xgb_model.predict(np.array([norm_sent_vector(sentence,ft) for sentence in df_shop[k].text]))) for k in df_shop.keys()],
        "precision": [precision_score(df_shop[k].rating, xgb_model.predict(np.array([norm_sent_vector(sentence,ft) for sentence in df_shop[k].text]))) for k in df_shop.keys()],
        "positif_count": [df_shop[k].loc[df_shop[k].rating == 1].text.shape[0] for k in df_shop.keys()],
        "negatif_count": [df_shop[k].loc[df_shop[k].rating == 0].text.shape[0] for k in df_shop.keys()],
        "total_review": [df_shop[k].text.shape[0] for k in df_shop.keys()], 
        "positif_percentage": [df_shop[k].loc[df_shop[k].rating == 1].text.shape[0] / df_shop[k].text.shape[0] * 100 for k in df_shop.keys()],
        "negatif_percentage": [df_shop[k].loc[df_shop[k].rating == 0].text.shape[0] / df_shop[k].text.shape[0] * 100 for k in df_shop.keys()]
    })
    
    # show final data
    st.dataframe(rank_shop.sort_values(by="f1_score", ascending=False), use_container_width=True)
    
