import pandas as pd
from gensim.models import Word2Vec
import re
import numpy as np
from difflib import SequenceMatcher


def clean_ingredient_text(text):
    if pd.isna(text):
        return []
    text = re.sub(r'\([^)]*\)', '', text)
    text = text.lower()

    ingredients = [ingredient.strip() for ingredient in text.split(',')]

    ingredients_cleaned = []
    for ingredient in ingredients:
        # Split on spaces and hyphens
        words = re.split(r'[\s\-]+', ingredient)
        ingredients_cleaned.extend([w for w in words if len(w) > 2])

    return ingredients_cleaned


def fuzzy_match_ingredients(target_list, product_list, threshold=0.8):
    """Find fuzzy matches between target ingredients and product ingredients."""
    matches = set()
    for target in target_list:
        for product in product_list:
            similarity = SequenceMatcher(None, target.lower(), product.lower()).ratio()
            if similarity >= threshold:
                matches.add(product)
    return matches


def prepare_word_dict(df_ingredients, df_products):
    dictionary = []

    for text in df_ingredients['Ingredients_to_Use'].dropna():
        tokens = clean_ingredient_text(text)
        if tokens:
            dictionary.append(tokens)

    for text in df_products['ingredients'].dropna():
        tokens = clean_ingredient_text(text)
        if tokens:
            dictionary.append(tokens)

    return dictionary


def train_word2vec(word_dict):
    model = Word2Vec(
        sentences=word_dict,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=20
    )
    return model


def sentence_embeddings(phrase: str, model: Word2Vec) -> np.ndarray:
    words = clean_ingredient_text(phrase)
    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def pre_process_data():
    df_ingredients = pd.read_csv("dermatology_ingredients.csv")
    df_products = pd.read_csv("cosmetic_p.csv")

    print("Data loaded successfully")
    print("Training Word2Vec model:")

    corpus = prepare_word_dict(df_ingredients, df_products)
    model = train_word2vec(corpus)

    print(f"Corpus size: {len(corpus)} documents")
    print(f"Vocabulary size: {len(model.wv)}")

pre_process_data()