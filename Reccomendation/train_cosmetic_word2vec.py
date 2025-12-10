
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
    corpus = []

    for item in dataset:
        tokens = clean_text(item['description'])
        if tokens:
            corpus.append(tokens)

    return corpus


def train_word2vec_cosmetic(save_path: str = "word2vec_cosmetic.model") -> Word2Vec:
   
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
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Training new model...")
        return train_word2vec_cosmetic(model_path)

    print(f"Loading model from {model_path}")
    return Word2Vec.load(model_path)


def get_embedding(phrase: str, model: Word2Vec) -> np.ndarray:

    words = clean_text(phrase)
    vectors = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)



