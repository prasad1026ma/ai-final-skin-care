# AI has been used to make the Evaluation output clearer

import pandas as pd
import numpy as np
import time
from unified_recommendation import UnifiedSkincareRecommendationEngine
from data_cleaning import clean_ingredient_text, fuzzy_match_ingredients


class EvaluationMetrics:

    def __init__(self):
        self.total_recommendations = 0
        self.beneficial_matches = 0
        self.harmful_matches = 0
        self.inference_times = []
        self.all_recommendations = []
        self.condition_stats = {}


class UnifiedRecommendationEvaluator:

    def __init__(self, products_path: str, conditions_path: str):
        self.engine = UnifiedSkincareRecommendationEngine(products_path, conditions_path)
        self.metrics = EvaluationMetrics()

    def analyze_product_match(self, recommendation, product_ingredients, beneficial_list, harmful_list):
        """Analyze ingredient matches for a single product recommendation using fuzzy matching (80% threshold)."""
        # Use fuzzy matching instead of exact matching
        beneficial_match = fuzzy_match_ingredients(beneficial_list, list(product_ingredients), threshold=0.8)
        harmful_match = fuzzy_match_ingredients(harmful_list, list(product_ingredients), threshold=0.8)

        return {
            'beneficial_match': beneficial_match,
            'beneficial_count': len(beneficial_match),
            'harmful_match': harmful_match,
            'harmful_count': len(harmful_match)
        }

    def print_recommendation_details(self, rank, product_name, rec, match_info, beneficial_total, harmful_total):
        print(f"\n      {rank}. {product_name}")
        print(f"         Price: ${rec['price']:.2f} | Rating: {rec['rating']:.1f}/5")
        print(f"         Overall Score: {rec['score']:.3f}")
        print(f"         Beneficial Similarity: {rec['beneficial_similarity']:.3f}")
        print(f"         Harmful Similarity: {rec['harmful_similarity']:.3f}")
        print(f"         Beneficial matches: {match_info['beneficial_count']}/{beneficial_total}")

        if match_info['beneficial_match']:
            sample = list(match_info['beneficial_match'])[:2]
            print(f"            Found: {', '.join(sample)}")

        print(f"         Harmful matches: {match_info['harmful_count']}/{harmful_total}")
        if match_info['harmful_match']:
            sample = list(match_info['harmful_match'])[:2]
            print(f"            Found: {', '.join(sample)}")



    def evaluate_condition(self, condition: str):
        """Evaluate recommendations for a single condition."""
        print(f"\n{condition}")
        print("-" * 80)

        try:
            beneficial_text, harmful_text = self.engine.get_condition_ingredients(condition)
            beneficial_list = clean_ingredient_text(beneficial_text)
            harmful_list = clean_ingredient_text(harmful_text)

            start_time = time.time()
            recommendations = self.engine.recommend(condition, top_n=3)
            inference_time = time.time() - start_time
            self.metrics.inference_times.append(inference_time)

            if recommendations:
                print(f"   Generated {len(recommendations)} recommendations in {inference_time:.3f}s")

            print(f"   Beneficial ingredients to match: {len(beneficial_list)}")
            print(f"   Harmful ingredients to avoid: {len(harmful_list)}")
            print(f"\n   TOP 3 RECOMMENDATIONS (using 80% fuzzy matching):")

            # Analyze each recommendation
            for i, rec in enumerate(recommendations, 1):
                product_name = f"{rec['brand']} - {rec['name']}"
                product = self.engine.products_df.iloc[rec['index']]
                product_ingredients = clean_ingredient_text(product['ingredients'])

                match_info = self.analyze_product_match(rec, product_ingredients, beneficial_list, harmful_list)
                self.print_recommendation_details(i, product_name, rec, match_info, len(beneficial_list), len(harmful_list))

                # Update metrics
                self.metrics.total_recommendations += 1
                if match_info['beneficial_count'] > 0:
                    self.metrics.beneficial_matches += 1
                if match_info['harmful_count'] > 0:
                    self.metrics.harmful_matches += 1

                # Store recommendation data
                self.metrics.all_recommendations.append({
                    'condition': condition,
                    'rank': i,
                    'product': product_name,
                    'score': rec['score'],
                    'beneficial_match_count': match_info['beneficial_count'],
                    'harmful_match_count': match_info['harmful_count'],
                    'beneficial_total': len(beneficial_list),
                    'harmful_total': len(harmful_list)
                })

        except Exception as e:
            print(f"   ✗ Error: {e}")




    def print_overall_stats(self):
        print(f"   Total recommendations: {self.metrics.total_recommendations}")
        print(f"   Conditions evaluated: {len(self.engine.supported_conditions)}")

        if self.metrics.total_recommendations > 0:
            print(f"   Recommendations per condition: {self.metrics.total_recommendations // len(self.engine.supported_conditions)}")
            beneficial_rate = (self.metrics.beneficial_matches / self.metrics.total_recommendations * 100)
            harmful_rate = (self.metrics.harmful_matches / self.metrics.total_recommendations * 100)
        else:
            beneficial_rate = 0
            harmful_rate = 0

        print(f"\nBENEFICIAL INGREDIENT MATCHING:")
        print(f"   Recommendations with beneficial ingredients: {self.metrics.beneficial_matches}/{self.metrics.total_recommendations}")
        print(f"   Match rate: {beneficial_rate:.1f}%")

        print(f"\nHARMFUL INGREDIENT AVOIDANCE:")
        print(f"   Recommendations with harmful ingredients: {self.metrics.harmful_matches}/{self.metrics.total_recommendations}")
        print(f"   Avoidance rate: {100 - harmful_rate:.1f}%")

        quality_score = (beneficial_rate - harmful_rate) / 100
        print(f"\nQUALITY SCORE: {quality_score:.2f}/1.00")




    def calculate_condition_stats(self):
        conditions = self.engine.supported_conditions

        for condition in conditions:
            cond_recs = [r for r in self.metrics.all_recommendations if r['condition'] == condition]
            if cond_recs:
                avg_score = np.mean([r['score'] for r in cond_recs])
                beneficial_rate = (sum(1 for r in cond_recs if r['beneficial_match_count'] > 0) / len(cond_recs) * 100)
                harmful_rate = (sum(1 for r in cond_recs if r['harmful_match_count'] > 0) / len(cond_recs) * 100)

                self.metrics.condition_stats[condition] = {
                    'avg_score': avg_score,
                    'beneficial_rate': beneficial_rate,
                    'harmful_rate': harmful_rate
                }



    def print_condition_performance(self):
        print("\n\nCONDITION-SPECIFIC PERFORMANCE")
        print("-" * 80)

        for condition, stats in self.metrics.condition_stats.items():
            print(f"\n{condition}")
            print(f"   Avg recommendation score: {stats['avg_score']:.3f}")
            print(f"   Beneficial match rate: {stats['beneficial_rate']:.1f}%")
            print(f"   Harmful match rate: {stats['harmful_rate']:.1f}%")



    def print_best_worst_conditions(self):
        """Print top and bottom performing conditions."""
        print("\n\nTOP AND BOTTOM PERFORMING CONDITIONS")
        print("-" * 80)

        quality_by_condition = {
            c: stats['beneficial_rate'] - stats['harmful_rate']
            for c, stats in self.metrics.condition_stats.items()
        }

        best = sorted(quality_by_condition.items(), key=lambda x: x[1], reverse=True)[:3]
        worst = sorted(quality_by_condition.items(), key=lambda x: x[1])[:3]

        print("\nTOP 3 BEST:")
        for i, (cond, quality) in enumerate(best, 1):
            print(f"   {i}. {cond} - Quality: {quality:.1f}%")

        print("\nTOP 3 WORST:")
        for i, (cond, quality) in enumerate(worst, 1):
            print(f"   {i}. {cond} - Quality: {quality:.1f}%")

    def evaluate(self):
        conditions = self.engine.supported_conditions

        # Evaluate each condition
        for condition in conditions:
            self.evaluate_condition(condition)

        # Print results
        self.print_overall_stats()
        self.calculate_condition_stats()
        self.print_condition_performance()
        self.print_best_worst_conditions()


def evaluate_recommendations():
    evaluator = UnifiedRecommendationEvaluator(
        products_path='cosmetic_p.csv',
        conditions_path='dermatology_ingredients.csv'
    )
    evaluator.evaluate()

if __name__ == "__main__":
    evaluate_recommendations()
