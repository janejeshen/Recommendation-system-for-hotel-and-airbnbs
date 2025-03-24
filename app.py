import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64


# Setting streamlit app layout
st.set_page_config(page_title='Accommodation Recommender', page_icon='🏨', layout="wide", initial_sidebar_state="auto")

# function to set up background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

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

# Setting background image
set_png_as_page_bg(r'background.jpg')

# Caching the Word2Vec model to avoid reloading
@st.cache_resource
def load_word2vec_model():
    return Word2Vec.load("word2vec.model")

# Caching the preprocessed data to avoid reloading
@st.cache_data
def load_data():
    # Loading preprocessed data
    df = pd.read_csv("data_2\Airbnb_Data_Preprocessed.csv")  
    
    # Combining text columns into a single text column
    if 'description_amenities' not in df.columns:
        df['description_amenities'] = df['description'] + " " + df['amenities'] + " " + df['name']
    
    # Handling missing values and type casting the combined text column
    df['description_amenities'] = df['description_amenities'].fillna("").astype(str)
    
    # Tokenizing the text column
    df['tokens'] = df['description_amenities'].apply(lambda x: x.split())
    
    return df

# Generating document embeddings by averaging word embeddings
def get_document_embedding(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Precomputing embeddings for all listings and saving to a file
def precompute_embeddings(df, model):
    embeddings_file = "precomputed_embeddings.npy"
    if not os.path.exists(embeddings_file):
        embeddings = np.array([get_document_embedding(tokens, model) for tokens in df['tokens']])
        np.save(embeddings_file, embeddings)
    else:
        embeddings = np.load(embeddings_file, allow_pickle=True)
    return embeddings

# Loading  Word2Vec model 
word2vec_model = load_word2vec_model()

# Loading preprocessed dataset
df = load_data()

# Precomputing embeddings for all listings
embeddings = precompute_embeddings(df, word2vec_model)
df['embedding'] = list(embeddings) 

# Caching filtered data by city to avoid recomputing
@st.cache_data
def filter_data_by_city(df, city):
    return df[df['city'].str.lower() == city.lower()]

# Getting recommendations based on user input
def get_recommendations_from_input(user_input, price_range=None, city=None, accommodates=None, top_n=5):
    # Tokenizing user input
    user_tokens = user_input.split()
    
    # Generating embedding for user input
    user_embedding = get_document_embedding(user_tokens, word2vec_model).reshape(1, -1)
    
    # Filtering data by city if a specific city is selected
    if city and city != "All Cities":
        filtered_df = filter_data_by_city(df, city)
    else:
        filtered_df = df
    
    # Filtering by price range if provided
    if price_range:
        min_price, max_price = price_range
        filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]
    
    # Filtering by accommodates if provided
    if accommodates:
        filtered_df = filtered_df[filtered_df['accommodates'] >= accommodates]
    
    # Computing cosine similarity between user input and filtered listings
    filtered_embeddings = np.array(filtered_df['embedding'].tolist())
    similarities = cosine_similarity(user_embedding, filtered_embeddings).flatten()
    filtered_df['similarity'] = similarities
    
    # Sorting by similarity and return top N recommendations
    filtered_df = filtered_df.sort_values(by='similarity', ascending=False)
    return filtered_df.head(top_n)[['name', 'price', 'city', 'accommodates']]

# creating a sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select a section", ["About", "Predictions", "Help"])

# App sections
if app_mode == "Predictions":
    st.write('# Input Section')
    st.write('#### Please describe your ideal accommodation.')
    
    # User input fields
    user_input = st.text_input("Describe your ideal accommodation (e.g., 'cozy apartment near the beach'):")
    price_range = st.slider("Select your price range ($):", 0, 500, (50, 200))
    # Adding "All Cities" option to the city selectbox
    city_options = ["All Cities"] + sorted(df['city'].unique().tolist())
    city = st.selectbox('City', city_options)
    accommodates = st.number_input("Number of people it should accommodate (optional):", min_value=1, value=2)
    top_n = st.number_input("Number of recommendations:", min_value=1, max_value=10, value=5)  # Default is 5

    if st.button("Get Recommendations"):
        if user_input:
            with st.spinner("Fetching recommendations..."):
                recommendations = get_recommendations_from_input( user_input,  price_range=price_range,  city=city, accommodates=accommodates, top_n=top_n)
                st.write("### Recommended Accommodations:")
                st.dataframe(recommendations)
        else:
            st.warning("Please provide a description of your ideal accommodation.")

elif app_mode == "About":
    st.header('🏨Accommodation Recommender In USA')
    st.write('### Project Overview')
    st.write('##### ⚠️This is for learning purposes‼️')
    st.markdown("""
        This project focuses on developing a recommendation tool that suggests accommodations based on user preferences such as location, price, and amenities. 
        The system uses advanced machine learning and natural language processing techniques to provide personalized recommendations, 
        addressing the limitations of traditional systems that struggle with unfamiliar users or insufficient data. 
        The goal is to simplify the search process and enhance the travel experience.
    """)
    st.write('### Objective')
    st.markdown("To develop a feature-based recommendation system for hotels and Airbnbs that matches users with their ideal accommodations.")

elif app_mode == "Help":
    st.markdown('## Help')
    st.markdown("This app is designed to recommend hotels and Airbnbs in the USA. Here's a quick guide to help you use this app:")
    st.write('###### To use this app, simply follow these steps:')
    st.markdown("""
        - Fill in the input form with your preferences (e.g., location, amenities, price range).
        - Click the **Get Recommendations** button to receive personalized accommodation suggestions.
        - The app will display a list of recommended properties based on your input.
    """)
    st.markdown("If you encounter any issues or have questions, feel free to reach out using the contact information below.")
    
    with st.container():
        st.write("### Contact Information")
        st.write("##### Email: janenjuguna550@gmail.com")
        st.write("##### Phone: +254114180510")
        st.write("##### Address: Nairobi, Kenya")
        st.write('##### GitHub: [janejeshen](https://github.com/janejeshen)')

