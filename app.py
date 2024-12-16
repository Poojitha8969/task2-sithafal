from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import torch
from bs4 import BeautifulSoup
import requests

# Initialize Flask app
app = Flask(__name__)

# Initialize the embedding model (using DistilBERT)
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# FAISS index initialization
def create_faiss_index(dimension):
    return faiss.IndexFlatL2(dimension)

# Create a simple FAISS index and initialize data storage paths
index = create_faiss_index(768)  # Embedding dimension
metadata = []
os.makedirs("data", exist_ok=True)

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])
    return text_content

# Function to convert text into embeddings
def convert_to_embeddings(text):
    return model.encode(text)

# Store embeddings in FAISS
def store_embeddings(embeddings, metadata_list):
    embeddings = np.array(embeddings)
    index.add(embeddings)  # Adding embeddings to FAISS index
    global metadata
    metadata = metadata_list  # Store metadata in global variable
    np.save(os.path.join("data", "embeddings.npy"), embeddings)  # Save embeddings
    with open(os.path.join("data", "metadata.txt"), "w") as f:
        for item in metadata_list:
            f.write(f"{item}\n")

# Query Handling
def query_to_embeddings(query):
    return model.encode(query)

def retrieve_relevant_chunks(query_embeddings, top_k=5):
    distances, indices = index.search(query_embeddings, top_k)
    return indices, distances

def get_metadata(indices):
    return [metadata[i] for i in indices.flatten()]

def generate_response(query, relevant_chunks):
    # Here, you can use a smaller language model or a rule-based approach to generate responses.
    # For example, you could concatenate the relevant chunks and return them as the response.
    relevant_text = [chunk['chunk'] for chunk in relevant_chunks]
    response = "\n".join(relevant_text)
    return response

# Main route for the app
@app.route('/')
def home():
    return render_template('index.html')

# API for handling website ingestion
@app.route('/ingest', methods=['POST'])
def ingest():
    url = request.form['url']
    text = scrape_website(url)
    chunks = text.split("\n")
    embeddings = [convert_to_embeddings(chunk) for chunk in chunks]
    metadata_list = [{"url": url, "chunk": chunk} for chunk in chunks]
    store_embeddings(embeddings, metadata_list)
    return jsonify({"message": "Website content ingested successfully!"})

# API for handling user query
@app.route('/query', methods=['POST'])
def query():
    try:
        user_query = request.form['query']
        query_embeddings = query_to_embeddings(user_query).reshape(1, -1)
        indices, distances = retrieve_relevant_chunks(query_embeddings)
        relevant_chunks = get_metadata(indices.flatten())

        # Generate the response
        response = generate_response(user_query, relevant_chunks)

        return jsonify({"response": response})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": "Error occurred while processing your request."}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)