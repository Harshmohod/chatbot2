import pandas as pd
import gradio as gr
import spacy
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import json

# Load data
df = pd.read_csv("netflix_titles.csv")
df.fillna("", inplace=True)

# Clean columns
df.columns = [col.lower().strip() for col in df.columns]

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all titles + descriptions
texts = df["title"] + " " + df["description"]
embeddings = model.encode(texts.tolist(), show_progress_bar=True)

# Define helper for filter extraction
def extract_filters(user_query):
    doc = nlp(user_query.lower())
    filters = {
        "type": None,
        "country": None,
        "listed_in": None,
        "release_year": None,
    }

    # Find year
    year_match = re.search(r"\b(19|20)\d{2}\b", user_query)
    if year_match:
        filters["release_year"] = year_match.group()

    for ent in doc.ents:
        if ent.label_ == "GPE":
            filters["country"] = ent.text.title()

    # Genre & Type manual match
    genre_keywords = {
        "action", "comedy", "drama", "horror", "romance", "thriller",
        "documentary", "sci-fi", "crime", "kids", "family", "anime"
    }
    for token in doc:
        if token.text.lower() in genre_keywords:
            filters["listed_in"] = token.text.title()

    if "movie" in user_query:
        filters["type"] = "Movie"
    elif "show" in user_query or "series" in user_query:
        filters["type"] = "TV Show"

    return filters

# Filter function
def filter_data(filters):
    temp = df.copy()
    if filters["type"]:
        temp = temp[temp["type"].str.lower() == filters["type"].lower()]
    if filters["country"]:
        temp = temp[temp["country"].str.contains(filters["country"], case=False, na=False)]
    if filters["listed_in"]:
        temp = temp[temp["listed_in"].str.contains(filters["listed_in"], case=False, na=False)]
    if filters["release_year"]:
        temp = temp[temp["release_year"].astype(str) == filters["release_year"]]
    return temp

# Mistral prompt builder
def build_prompt(query, results):
    if results.empty:
        return f"User asked: '{query}'\nThere were no matching results in the Netflix database."

    top_3 = results.head(3).to_dict(orient="records")
    prompt = f"You are a helpful Netflix assistant. User asked: \"{query}\".\n"
    prompt += "Based on the database, here are some matching titles:\n\n"
    for item in top_3:
        prompt += f"- **{item['title']}** ({item['release_year']}): {item['description']}\n"
    prompt += "\nGive a short natural reply to the user."
    return prompt

# Call Mistral using Ollama
def query_mistral(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", "--", prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error calling Mistral: {e}"

# Main chatbot function
def chat(query):
    filters = extract_filters(query)
    filtered = filter_data(filters)

    # Semantic similarity fallback if no filter result
    if filtered.empty:
        query_embed = model.encode([query])
        sims = cosine_similarity(query_embed, embeddings)[0]
        top_idx = np.argsort(sims)[-5:][::-1]
        filtered = df.iloc[top_idx]

    prompt = build_prompt(query, filtered)
    response = query_mistral(prompt)
    return response

# Gradio UI
iface = gr.Interface(fn=chat, inputs="text", outputs="text", title="🎬 Netflix Chatbot with Mistral")

if __name__ == "__main__":
    iface.launch()
