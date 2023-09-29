import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from stqdm import stqdm

st.set_page_config(layout='wide')
st.title('AniRec!')
st.markdown('''
            Get anime recommendations **based on the last anime you watched**. It'll automatically find similar animes
            for you !
            ''')

st.markdown('''
            <style>
                footer{
                visibility: hidden;
                }
            
                h1#anirec{
                    color: #af69ef;
                    font-size: 3rem;

                }
            </style>
            ''', unsafe_allow_html=True)



animes = pd.read_csv(r'F:\datasets(AI stuff)\arecomai\animes_cleaned.csv')

def anime_recommend(title, cosine_sim, dataframe):
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    
    anime_index = indices[title]

    similarity_scores = pd.DataFrame(cosine_sim[anime_index], columns=["similarity"])

    anime_indices = similarity_scores.sort_values("similarity", ascending=False)[1:30].index
    recomm = dataframe.iloc[anime_indices].join(similarity_scores)
    recomm['matched_with'] = title

    return recomm[['title', 'tags', 'similarity', 'score', 'episodes', 'matched_with']]

@st.cache_data
def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english', max_features = 5000)
    tfidf_matrix = tfidf.fit_transform(dataframe['tags'].values.astype('U'))
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim


st.sidebar.title('Select Anime')
st.sidebar.write('Select the animes and the AI would recommend based on that.')

st.sidebar.divider()

selection = st.sidebar.selectbox('Animes', animes['title'].values)
st.sidebar.caption('Select the anime you previously watched')

n_recom = st.sidebar.slider('No. of recommendations', min_value=5, max_value=30, value=10)
st.sidebar.caption('The number of recommendations you want')

if st.sidebar.button('Recommend'):
    
    st.divider()
    cossim = calculate_cosine_sim(animes)
    recomm = anime_recommend(selection, cossim, animes)
    st.write('Recommended animes')
    st.write(recomm[:n_recom])
    st.balloons()

