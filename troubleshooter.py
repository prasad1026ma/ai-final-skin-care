import pandas as pd
import numpy as np
import time
from Reccomendation.recommendation import SkincareRecommendationEngine
from Reccomendation.data_cleaning import clean_ingredient_text
from Classification.classification_pipeline import run_classification_pipeline

def print_reccomendation(recommendations):
    for i, rec in enumerate(recommendations, 1):
        product_name = f"{rec['brand']} - {rec['name']}"
        print(f"\n      {i}. {product_name}")
        print(f"         Price: ${rec['price']:.2f} | Rating: {rec['rating']:.1f}/5")
        print(f"         Overall Score: {rec['score']:.3f}")
        print(f"         Beneficial Similarity: {rec['beneficial_similarity']:.3f}")
        print(f"         Harmful Similarity: {rec['harmful_similarity']:.3f}")

def run_troubleshooter():
    image_path = input("Enter the path to the skin lesion image: ").strip()
    input_size = 224

    predicted_class, confidence = run_classification_pipeline(image_path, input_size)
    print(f"\nPredicted Skin Lesion: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    engine = SkincareRecommendationEngine(
            products_path='Reccomendation/cosmetic_p.csv',
            conditions_path='Reccomendation/dermatology_ingredients.csv'
    )
    try:
        recommendations = engine.recommend_top_3(predicted_class)
        print_reccomendation(recommendations)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_troubleshooter()



