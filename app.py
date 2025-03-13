import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
import pickle as pkl
import base64
import warnings
warnings.filterwarnings('error')
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

st.set_page_config(page_title='Accommodation Recommender', page_icon='🏨', layout="wide", initial_sidebar_state="auto")

# Defining CSS style to change markdown text color to black
black_text = """
<style>
body {
    color: black;
}
</style>
"""

# Appling the style to your Streamlit app
st.markdown(black_text, unsafe_allow_html=True)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

import base64

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
        .stApp {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: scroll;
        }
        .stApp::after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.6);
            z-index: -1;
        }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg('dwiinshito--lDNCLbQi9g-unsplash (1).jpg')

# Creating a side bar
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select a section", ["About", "Predictions",'Help'])

# Loading the model
@st.cache_resource
def load_word2vec_model():
    model = Word2Vec.load("word2vec.model")
    return model

# Loading dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data_2/Airbnb_Data.csv")
    return df

# Generating document embeddings by averaging word embeddings
def get_document_embedding(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    

# Getting recommendations based on user input
def get_recommendations_from_input(user_input, price_range=None, top_n=5):
    # Tokenizing user input
    user_tokens = user_input.split()
    
    # Generating embedding for user input
    user_embedding = get_document_embedding(user_tokens, word2vec_model)
    
    # Computing cosine similarity between user input and all listings
    similarities = df['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])
    df['similarity'] = similarities
    
    # Sorting by similarity
    df_sorted = df.sort_values(by='similarity', ascending=False)
    
    # Filtering by price range if required
    if price_range:
        min_price, max_price = price_range
        recommendations = df_sorted[(df_sorted['price'] >= min_price) & (df_sorted['price'] <= max_price)]
    else:
        recommendations = df_sorted
    
    # Returning top N recommendations
    return recommendations.head(top_n)[['name', 'price', 'description','amenities']]


# Loading the Word2Vec model and dataset
word2vec_model = load_word2vec_model()
df = load_data()

# Generating document embeddings for all listings
df['embedding'] = df['tokens'].apply(lambda x: get_document_embedding(x, word2vec_model))