import pandas as pd
import numpy as np
from data_cleaning import (
    train_word2vec,
    prepare_word_dict,
    sentence_embeddings
)

# FASTTEXT MODEL TRAINING AND RECOMMENDATION ENGINE
class SkincareRecommendationEngine:
  
    # INITIALIZATION
    def __init__(self, products_path: str, conditions_path: str):
  
        self.products_df = pd.read_csv(products_path)
        self.conditions_df = pd.read_csv(conditions_path)

        corpus = prepare_word_dict(self.conditions_df, self.products_df)
        self.model = train_word2vec(corpus)

        self.condition_embeddings = {}
        self.harmful_embeddings = {}

        for _, row in self.conditions_df.iterrows():
            condition = row['Condition']
            beneficial_text = row['Ingredients_to_Use']
            harmful_text = row['Ingredients_to_Avoid']

            self.condition_embeddings[condition] = sentence_embeddings(
                beneficial_text, self.model
            )
            self.harmful_embeddings[condition] = sentence_embeddings(
                harmful_text, self.model
            )

        self.product_embeddings = []
        for _, row in self.products_df.iterrows():
            product_embedding = sentence_embeddings(row['ingredients'], self.model)
            self.product_embeddings.append(product_embedding)

        self.product_embeddings = np.array(self.product_embeddings)

        self.supported_conditions = list(self.conditions_df['Condition'].unique())

    # COSINE SIMILARITY CALCULATION
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return max(0.0, min(1.0, similarity))
    
    # GET BENEFICIAL AND HARMFUL INGREDIENTS FOR A CONDITION
    def get_condition_ingredients(self, condition: str) -> tuple:

        condition_row = self.conditions_df[
            self.conditions_df['Condition'] == condition
        ]

        if condition_row.empty:
            raise ValueError(f"Condition '{condition}' not found. "
                           f"Supported: {', '.join(self.supported_conditions)}")

        beneficial = condition_row.iloc[0]['Ingredients_to_Use']
        harmful = condition_row.iloc[0]['Ingredients_to_Avoid']

        return beneficial, harmful

    # RECOMMEND TOP 3 PRODUCTS FOR A GIVEN CONDITION
    def recommend_top_3(self, condition: str) -> list:
        if condition not in self.condition_embeddings:
            raise ValueError(f"Condition '{condition}' not found. "
                           f"Supported: {', '.join(self.supported_conditions)}")

        beneficial_embedding = self.condition_embeddings[condition]
        harmful_embedding = self.harmful_embeddings[condition]

        beneficial_text, harmful_text = self.get_condition_ingredients(condition)

        beneficial_similarities = []
        harmful_similarities = []

        for product_embedding in self.product_embeddings:
            ben_sim = self.cosine_similarity(beneficial_embedding, product_embedding)
            beneficial_similarities.append(ben_sim)

            har_sim = self.cosine_similarity(harmful_embedding, product_embedding)
            harmful_similarities.append(har_sim)

        beneficial_similarities = np.array(beneficial_similarities)
        harmful_similarities = np.array(harmful_similarities)

        '''
        Scoring formula ************
        '''
        scores = (beneficial_similarities ** 2) * (1.0 - (harmful_similarities ** 2))

        top_3 = np.argsort(scores)[::-1][:3]

        recommendations = []
        for rank, idx in enumerate(top_3, 1):
            product = self.products_df.iloc[idx]
            recommendations.append({
                'rank': rank,
                'index': idx,
                'name': product['name'],
                'brand': product['brand'],
                'price': product['price'],
                'rating': product['rank'],
                'category': product['Label'],
                'score': scores[idx],
                'beneficial_similarity': beneficial_similarities[idx],
                'harmful_similarity': harmful_similarities[idx]
            })

        return recommendations

    def get_supported_conditions(self) -> list:
        return self.supported_conditions



# SAMPLE USAGE
def main():
    engine = SkincareRecommendationEngine(
        products_path='cosmetic_p.csv',
        conditions_path='dermatology_ingredients.csv'
    )

    test_conditions = ['Eczema', 'Psoriasis', 'Folliculitis']

    for condition in test_conditions:
        try:
            recommendations = engine.recommend_top_3(condition)
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
