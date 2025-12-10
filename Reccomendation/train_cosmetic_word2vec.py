"""
Train Word2Vec model using Hugging Face cosmetic-ingredients dataset.

This script provides an alternative to data_cleaning.pre_process_data()
for training Word2Vec on a richer set of ingredient descriptions.

Usage:
    from train_cosmetic_word2vec import load_trained_model
    model = load_trained_model()  # Loads the trained model
"""

import re
from pathlib import Path
from datasets import load_dataset
from gensim.models import Word2Vec
import numpy as np


def clean_text(text: str) -> list:
    """Clean and tokenize text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]


def load_cosmetic_dataset():
    """Load cosmetic-ingredients dataset from Hugging Face."""
    print("Loading cosmetic-ingredients dataset from Hugging Face...")
    dataset = load_dataset("yavuzyilmaz/cosmetic-ingredients")
    return dataset['train']


def prepare_corpus(dataset) -> list:
    """Prepare training corpus from dataset descriptions."""
    print("Preparing corpus from ingredient descriptions...")
    corpus = []

    for item in dataset:
        tokens = clean_text(item['description'])
        if tokens:
            corpus.append(tokens)

    return corpus


def train_word2vec_cosmetic(save_path: str = "word2vec_cosmetic.model") -> Word2Vec:
    """
    Train Word2Vec model on cosmetic-ingredients dataset.

    Args:
        save_path: Path to save the trained model

    Returns:
        Trained Word2Vec model
    """
    print("Training Word2Vec on cosmetic-ingredients dataset...")

    # Load and prepare data
    dataset = load_cosmetic_dataset()
    corpus = prepare_corpus(dataset)

    print(f"Corpus size: {len(corpus)} documents")

    # Train model
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        sg=1,
        epochs=10
    )

    print(f"Vocabulary size: {len(model.wv)}")

    # Save model
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model


def load_trained_model(model_path: str = "word2vec_cosmetic.model") -> Word2Vec:
    """
    Load a pre-trained Word2Vec model.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded Word2Vec model
    """
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Training new model...")
        return train_word2vec_cosmetic(model_path)

    print(f"Loading model from {model_path}")
    return Word2Vec.load(model_path)


def get_embedding(phrase: str, model: Word2Vec) -> np.ndarray:
    """
    Get embedding for a phrase using the Word2Vec model.

    Args:
        phrase: Text to embed
        model: Word2Vec model

    Returns:
        Embedding vector
    """
    words = clean_text(phrase)
    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)


if __name__ == "__main__":
    # Train and save model
    model = train_word2vec_cosmetic()

    # Test the model
    print("\n" + "="*50)
    print("Testing model on cosmetic terms:")
    print("="*50)

    test_terms = [
        "moisturizing ingredients",
        "soothing and hydrating",
        "antioxidant benefits",
        "gentle cleanser"
    ]

    for term in test_terms:
        embedding = get_embedding(term, model)
        print(f"\n'{term}': shape {embedding.shape}")
