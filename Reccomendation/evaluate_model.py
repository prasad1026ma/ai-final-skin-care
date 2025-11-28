import pandas as pd
import numpy as np
import time
from recommendation import SkincareRecommendationEngine
from data_cleaning import clean_ingredient_text


def evaluate_recommendations():
    # Initialize engine
    engine = SkincareRecommendationEngine(
        products_path='cosmetic_p.csv',
        conditions_path='dermatology_ingredients.csv'
    )

    conditions = engine.get_supported_conditions()

    all_recommendations = []
    beneficial_matches = 0
    harmful_matches = 0
    total_recommendations = 0
    inference_times = []

    # Evaluate each condition
    for condition in conditions:
        print(f"\n{condition}")
        print("-" * 80)

        try:
            beneficial_text, harmful_text = engine.get_condition_ingredients(condition)
            beneficial_set = set(clean_ingredient_text(beneficial_text))
            harmful_set = set(clean_ingredient_text(harmful_text))

            recommendations = engine.recommend_top_3(condition, verbose=False)
            
            print(f"   Beneficial ingredients to match: {len(beneficial_set)}")
            print(f"   Harmful ingredients to avoid: {len(harmful_set)}")
            print(f"\n   TOP 3 RECOMMENDATIONS:")

            # Analyze each recommendation
            for i, rec in enumerate(recommendations, 1):
                product_name = f"{rec['brand']} - {rec['name']}"
                product = engine.products_df.iloc[rec['index']]
                product_ingredients = set(clean_ingredient_text(product['ingredients']))

                beneficial_match = beneficial_set & product_ingredients
                beneficial_match_count = len(beneficial_match)

                harmful_match = harmful_set & product_ingredients
                harmful_match_count = len(harmful_match)

                total_recommendations += 1
                if beneficial_match_count > 0:
                    beneficial_matches += 1
                if harmful_match_count > 0:
                    harmful_matches += 1

                print(f"\n      {i}. {product_name}")
                print(f"         Price: ${rec['price']:.2f} | Rating: {rec['rating']:.1f}/5")
                print(f"         Overall Score: {rec['score']:.3f}")
                print(f"         Beneficial Similarity: {rec['beneficial_similarity']:.3f}")
                print(f"         Harmful Similarity: {rec['harmful_similarity']:.3f}")
                print(f"         ✓ Beneficial matches: {beneficial_match_count}/{len(beneficial_set)}")
                if beneficial_match:
                    sample_matches = list(beneficial_match)[:2]
                    print(f"            Found: {', '.join(sample_matches)}")
                print(f"         ✗ Harmful matches: {harmful_match_count}/{len(harmful_set)}")
                if harmful_match:
                    sample_harmful = list(harmful_match)[:2]
                    print(f"            Found: {', '.join(sample_harmful)}")

                all_recommendations.append({
                    'condition': condition,
                    'rank': i,
                    'product': product_name,
                    'score': rec['score'],
                    'beneficial_match_count': beneficial_match_count,
                    'harmful_match_count': harmful_match_count,
                    'beneficial_total': len(beneficial_set),
                    'harmful_total': len(harmful_set)
                })

        except Exception as e:
            print(f"   ✗ Error: {e}")

    # Overall statistics
    print("EVALUATION SUMMARY")

    print(f"\nOVERALL METRICS:")
    print(f"   Total recommendations: {total_recommendations}")
    print(f"   Conditions evaluated: {len(conditions)}")
    print(f"   Recommendations per condition: {total_recommendations // len(conditions)}")

    print(f"\nBENEFICIAL INGREDIENT MATCHING:")
    beneficial_rate = (beneficial_matches / total_recommendations * 100) if total_recommendations > 0 else 0
    print(f"   Recommendations with beneficial ingredients: {beneficial_matches}/{total_recommendations}")
    print(f"   Match rate: {beneficial_rate:.1f}%")

    print(f"\nHARMFUL INGREDIENT AVOIDANCE:")
    harmful_rate = (harmful_matches / total_recommendations * 100) if total_recommendations > 0 else 0
    print(f"   Recommendations with harmful ingredients: {harmful_matches}/{total_recommendations}")
    print(f"   Avoidance rate: {100 - harmful_rate:.1f}%")

    quality_score = (beneficial_rate - harmful_rate) / 100
    print(f"\nQUALITY SCORE: {quality_score:.2f}/1.00")

    # Condition-specific performance
    print("CONDITION-SPECIFIC PERFORMANCE")
    print("-"*80)

    condition_stats = {}
    for condition in conditions:
        cond_recs = [r for r in all_recommendations if r['condition'] == condition]
        if cond_recs:
            avg_score = np.mean([r['score'] for r in cond_recs])
            beneficial_rate_cond = (
                sum(1 for r in cond_recs if r['beneficial_match_count'] > 0) / len(cond_recs) * 100
            )
            harmful_rate_cond = (
                sum(1 for r in cond_recs if r['harmful_match_count'] > 0) / len(cond_recs) * 100
            )

            condition_stats[condition] = {
                'avg_score': avg_score,
                'beneficial_rate': beneficial_rate_cond,
                'harmful_rate': harmful_rate_cond
            }

            print(f"\n{condition}")
            print(f"   Avg recommendation score: {avg_score:.3f}")
            print(f"   Beneficial match rate: {beneficial_rate_cond:.1f}%")
            print(f"   Harmful match rate: {harmful_rate_cond:.1f}%")

    # Best and worst performing conditions
    print("TOP AND BOTTOM PERFORMING CONDITIONS")

    quality_by_condition = {
        c: stats['beneficial_rate'] - stats['harmful_rate']
        for c, stats in condition_stats.items()
    }

    print("-" * 80)

    best_conditions = sorted(quality_by_condition.items(), key=lambda x: x[1], reverse=True)[:3]
    worst_conditions = sorted(quality_by_condition.items(), key=lambda x: x[1])[:3]

    print("\nTOP 3 BEST PERFORMING:")
    for i, (cond, quality) in enumerate(best_conditions, 1):
        print(f"   {i}. {cond} - Quality: {quality:.1f}%")

    print("\nTOP 3 NEEDS IMPROVEMENT:")
    for i, (cond, quality) in enumerate(worst_conditions, 1):
        print(f"   {i}. {cond} - Quality: {quality:.1f}%")

if __name__ == "__main__":
    evaluate_recommendations()
