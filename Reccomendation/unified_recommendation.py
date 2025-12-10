import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
from data_cleaning import clean_ingredient_text, pre_process_data

# Choose your embedding model:
#   - "csv" (Word2Vec trained on your CSV data)
#   - "cosmetic" (Word2Vec trained on Hugging Face cosmetic-ingredients))
#   - "scibert" 
#   - "biobert" 
#   - "chembert" 
EMBEDDING_MODEL = "cosmetic"

TRANSFORMER_MODEL_NAMES = {
    "scibert": "allenai/scibert_scivocab_uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "chembert": "DeepChem/ChemBERTa-77M-MLM"
}

'''
Provider for the getting different embedding models for the recommenation. 
'''
class EmbeddingProvider:
    def __init__(self, model_type: str, products_df: pd.DataFrame, conditions_df: pd.DataFrame):
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = None

        if model_type == "csv":
            self._init_word2vec_csv()
        elif model_type == "cosmetic":
            self._init_word2vec_cosmetic()
        else:
            self._init_transformer(model_type)

    def _init_word2vec_csv(self):
        self.model = pre_process_data()

    def _init_word2vec_cosmetic(self):
        try:
            from train_cosmetic_word2vec import load_trained_model
            self.model = load_trained_model()
        except ImportError:
            self.model = pre_process_data()

    def _init_transformer(self, model_type: str):
        model_name = TRANSFORMER_MODEL_NAMES.get(model_type)
        if not model_name:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"Loading {model_type.upper()} model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text: str) -> np.ndarray:
        if pd.isna(text) or not text:
            if self.model_type in ["csv", "cosmetic"]:
                return np.zeros(100)
            else:
                return np.zeros(768)

        if self.model_type in ["csv", "cosmetic"]:
            return self._get_word2vec_embedding(text)
        else:
            return self._get_transformer_embedding(text)

    def _get_word2vec_embedding(self, text: str) -> np.ndarray:
        words = clean_ingredient_text(text)
        vectors = []
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(100) 

    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return embedding

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
Cosine Similarity Function 
'''
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
    return max(0.0, min(1.0, similarity))

'''
Recommemdation Scoring
'''
def calculate_recommendation_score(beneficial_sim: np.ndarray, harmful_sim: np.ndarray) -> np.ndarray:
    return beneficial_sim * np.exp(-harmful_sim)

def apply_label_weights(scores: np.ndarray, products_df: pd.DataFrame) -> np.ndarray:
    weighted_scores = scores.copy()

    for idx in range(len(products_df)):
        label = products_df.iloc[idx]['Label']
        if label == 'Moisturizer':
            weighted_scores[idx] *= 1.3
        elif label == 'Treatment':
            weighted_scores[idx] *= 1.3

    return weighted_scores


def build_recommendation(rank: int, idx: int, scores: np.ndarray,
                       ben_sim: np.ndarray, harm_sim: np.ndarray,
                       products_df: pd.DataFrame) -> dict:
    product = products_df.iloc[idx]

    return {
        'rank': rank,
        'index': idx,
        'name': product['name'],
        'brand': product['brand'],
        'price': product['price'],
        'rating': product['rank'],
        'category': product['Label'],
        'score': scores[idx],
        'beneficial_similarity': ben_sim[idx],
        'harmful_similarity': harm_sim[idx]
    }

class UnifiedSkincareRecommendationEngine:

    def __init__(self, products_path: str, conditions_path: str):
        self.products_df = pd.read_csv(products_path)
        self.conditions_df = pd.read_csv(conditions_path)
        self.model_type = EMBEDDING_MODEL

        self.embedding_provider = EmbeddingProvider(self.model_type, self.products_df, self.conditions_df)
        self.condition_embeddings = {}
        self.harmful_embeddings = {}
        
        # Compute embeddings for conditions (including condition name + ingredients)
        for _, row in self.conditions_df.iterrows():
            condition = row['Condition']
            beneficial_text = row['Ingredients_to_Use']
            harmful_text = row['Ingredients_to_Avoid']

            # Get condition name embedding
            condition_embedding = self.embedding_provider.get_embedding(condition)

            # Get ingredient embeddings
            beneficial_ing_embedding = self.embedding_provider.get_embedding(beneficial_text)
            harmful_ing_embedding = self.embedding_provider.get_embedding(harmful_text)

            # Combine condition embedding with ingredient embeddings (weighted average)
            # 80% condition context, 20% ingredient specificity
            self.condition_embeddings[condition] = (0.8 * condition_embedding + 0.2 * beneficial_ing_embedding)
            self.harmful_embeddings[condition] = (0.8 * condition_embedding + 0.2 * harmful_ing_embedding)

        # Compute embeddings for products
        self.product_embeddings = []
        for _, row in self.products_df.iterrows():
            product_embedding = self.embedding_provider.get_embedding(row['ingredients'])
            self.product_embeddings.append(product_embedding)

        self.product_embeddings = np.array(self.product_embeddings)
        self.supported_conditions = list(self.conditions_df['Condition'].unique())

    def recommend(self, condition: str, top_n: int = 3) -> list:

        ben_embedding = self.condition_embeddings[condition]
        harm_embedding = self.harmful_embeddings[condition]

        ben_similarities = []
        harm_similarities = []

        for product_embedding in self.product_embeddings:
            ben_sim = cosine_similarity(ben_embedding, product_embedding)
            ben_similarities.append(ben_sim)

            harm_sim = cosine_similarity(harm_embedding, product_embedding)
            harm_similarities.append(harm_sim)

        ben_similarities = np.array(ben_similarities)
        harm_similarities = np.array(harm_similarities)

        scores = calculate_recommendation_score(ben_similarities, harm_similarities)

        scores = apply_label_weights(scores, self.products_df)

        top_n_indices = np.argsort(scores)[::-1][:top_n]

        recommendations = [
            build_recommendation(rank, idx, scores, ben_similarities, harm_similarities, self.products_df)
            for rank, idx in enumerate(top_n_indices, 1)
        ]

        return recommendations

    def get_condition_ingredients(self, condition: str) -> tuple:
        condition_row = self.conditions_df[self.conditions_df['Condition'] == condition]

        beneficial = condition_row.iloc[0]['Ingredients_to_Use']
        harmful = condition_row.iloc[0]['Ingredients_to_Avoid']

        return beneficial, harmful

def main():

    engine = UnifiedSkincareRecommendationEngine(
        products_path='cosmetic_p.csv',
        conditions_path='dermatology_ingredients.csv'
    )

    try:
        recommendations = engine.recommend('Psoriasis')
        print("\nTop recommendations for Psoriasis:")
        for rec in recommendations:
            print(f"  {rec['rank']}. {rec['brand']} - {rec['name']} (Score: {rec['score']:.3f})")
    except ValueError as e:
        print(f"Error: {e}")

   

if __name__ == "__main__":
    main()
