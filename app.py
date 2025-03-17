import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from langdetect import detect, DetectorFactory

# Set seed for consistent language detection
DetectorFactory.seed = 42

# Load spaCy English model
nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

# Function to detect if the language is English
def detect_language(text):
    try:
        return "en" if detect(text) == "en" else None
    except:
        return None

# Function to preprocess text
def preprocess_text(text):
    # Skip non-English text or missing values
    if pd.isna(text) or detect_language(text) is None:
        return ""
    
    # Parse the text with spaCy
    doc = nlp(text)
    
    # Lemmatize tokens and remove non-alphabetic tokens
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
    
    # Join lemmas back into a string
    return " ".join(lemmas)

# Load Word2Vec model
@st.cache_resource
def load_word2vec_model():
    model = Word2Vec.load("word2vec.model")
    return model

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data_2/Airbnb_Data.csv")
    
    # Handle missing values
    df['amenities'] = df['amenities'].fillna("")
    df['description'] = df['description'].fillna("")
    df['name'] = df['name'].fillna("")
    
    # Preprocess text columns
    df["amenities"] = [preprocess_text(text) for text in df["amenities"]]
    df["description"] = [preprocess_text(text) for text in df["description"]]
    df["name"] = [preprocess_text(text) for text in df["name"]]
    
    # Combine text features into one feature
    df["description_amenities"] = df["description"] + " " + df["amenities"] + " " + df["name"]
    
    # Tokenize the text column
    df['tokens'] = df['description_amenities'].apply(lambda x: x.split())
    
    return df

# Generate document embeddings by averaging word embeddings
def get_document_embedding(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Get recommendations based on user input
def get_recommendations_from_input(user_input, price_range=None, top_n=5):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    user_tokens = processed_input.split()
    
    # Generate embedding for user input
    user_embedding = get_document_embedding(user_tokens, word2vec_model)
    
    # Compute cosine similarity between user input and all listings
    similarities = df['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])
    df['similarity'] = similarities
    
    # Sort by similarity
    df_sorted = df.sort_values(by='similarity', ascending=False)
    
    # Filter by price range if provided
    if price_range:
        min_price, max_price = price_range
        df_sorted = df_sorted[(df_sorted['price'] >= min_price) & (df_sorted['price'] <= max_price)]
    
    # Handle missing values
    df_sorted['name'] = df_sorted['name'].fillna("Unnamed Listing")
    df_sorted['description'] = df_sorted['description'].fillna("No description available")
    df_sorted['amenities'] = df_sorted['amenities'].fillna("No amenities listed")
    
    # Return top N recommendations
    return df_sorted.head(top_n)[['name', 'price', 'description', 'amenities']]

# Load the Word2Vec model and dataset
word2vec_model = load_word2vec_model()
df = load_data()

# Generate document embeddings for all listings (if not already done)
if 'embedding' not in df.columns:
    df['embedding'] = df['tokens'].apply(lambda x: get_document_embedding(x, word2vec_model))

# Streamlit app layout
st.set_page_config(page_title='Accommodation Recommender', page_icon='🏨', layout="wide", initial_sidebar_state="auto")

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select a section", ["About", "Predictions", "Help"])

# App sections
if app_mode == "Predictions":
    st.write('# Input Section')
    st.write('#### Please describe your ideal accommodation.')
    
    # User input fields
    user_input = st.text_input("Describe your ideal accommodation (e.g., 'cozy apartment near the beach'):")
    price_range = st.slider("Select your price range ($):", 0, 500, (50, 200))
    top_n = st.number_input("Number of recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Get Recommendations"):
        if user_input:
            with st.spinner("Fetching recommendations..."):
                recommendations = get_recommendations_from_input(user_input, price_range, top_n)
                st.write("### Recommended Accommodations:")
                st.dataframe(recommendations)
        else:
            st.warning("Please provide a description of your ideal accommodation.")

elif app_mode == "About":
    st.header('Hotel and Airbnb Recommendation System in the USA')
    st.write('### Project Overview')
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



# # Setting seed for consistent language detection
# DetectorFactory.seed = 42

# # loading the English language model in spaCy
# nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])

# # Function for detecting if language used is english
# def detect_language(text):
#     try:
#         return "en" if detect(text) == "en" else None
#     except:
#         return None 

# # Function for preprocessing text
# def preprocess_text(text):
    
#     # Skipping non-English words and missing text
#     if pd.isna(text) or detect_language(text) is None:
#         return ""
    
#     # Parsing the text with spaCy
#     doc = nlp(text)
    
#     # Lemmatizing the tokens and removing non alphabetic tokens 
#     lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
    
#     # Joining the lemmas back into a string and returning it
#     return " ".join(lemmas)

# # applying the preprocess_text function to the text column
# df["amenities"] = [preprocess_text(text) for text in df["amenities"]]
# df["description"] = [preprocess_text(text) for text in df["description"]]
# df["name"] = [preprocess_text(text) for text in df["name"]]
# # Combining the text features into one feature
# df["description_amenities"] = df["description"] + " " + df["amenities"] + " " + df["name"]

# # Tokenizing the text column
# df['tokens'] = df['description_amenities'].apply(lambda x: x.split())
# # Training word2vec model
# word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
# # Generating document embeddings by averaging word embeddings
# def get_document_embedding(tokens, model):
#     vectors = [model.wv[word] for word in tokens if word in model.wv]
#     if len(vectors) > 0:
#         return np.mean(vectors, axis=0)
#     else:
#         return np.zeros(model.vector_size)
    
# # Generating document embeddings for all listings
# df['embedding'] = df['tokens'].apply(lambda x: get_document_embedding(x, word2vec_model))

# def get_recommendations_from_input(user_input, price_range=None, top_n=5):
#     # Tokenizing user input
#     user_tokens = user_input.split()
    
#     # Generating embedding for user input
#     user_embedding = get_document_embedding(user_tokens, word2vec_model)
    
#     # Computing cosine similarity between user input and all listings
#     similarities = df['embedding'].apply(lambda x: cosine_similarity([user_embedding], [x])[0][0])
#     df['similarity'] = similarities
    
#     # Sorting by similarity
#     df_sorted = df.sort_values(by='similarity', ascending=False)
    
#     # Filtering by price range if required
#     if price_range:
#         min_price, max_price = price_range
#         recommendations = df_sorted[(df_sorted['price'] >= min_price) & (df_sorted['price'] <= max_price)]
#     else:
#         recommendations = df_sorted
    
#     # **Fix missing values**
#     recommendations = recommendations.dropna(subset=['name'])  # Drop rows where 'name' is missing
#     recommendations['description'] = recommendations['description'].fillna("No description available")
#     recommendations['amenities'] = recommendations['amenities'].fillna("No amenities listed")

#     return recommendations.head(top_n)[['name', 'price', 'description', 'amenities']]

# user_input = "new york"
# price_range = (50, 200)

# recommendations = get_recommendations_from_input(user_input, price_range)

# print("Final Recommendations:")
# recommendations



# # Setting background image
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# import base64

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file) 
#     page_bg_img = '''
#     <style>
#         .stApp {
#             background-image: url("data:image/png;base64,%s");
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: scroll;
#         }
#         .stApp::after {
#             content: "";
#             position: absolute;
#             top: 0;
#             left: 0;
#             right: 0;
#             bottom: 0;
#             background-color: rgba(255, 255, 255, 0.6);
#             z-index: -1;
#         }
#     </style>
#     ''' % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# set_png_as_page_bg(r'C:\Users\janen\Documents\Research\Recommendation-system-for-hotel-and-airbnbs\background.jpg')