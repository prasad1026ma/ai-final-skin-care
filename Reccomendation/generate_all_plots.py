"""
Generate all visualizations for all conditions and all embedding models.

This script creates comprehensive visualizations of top 3 recommendations
for every skin condition using all available embedding models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
from datetime import datetime


def plot_top3_for_model(model_type: str, condition: str, method: str = 'tsne',
                        products_path: str = 'cosmetic_p.csv',
                        conditions_path: str = 'dermatology_ingredients.csv'):
    """
    Plot top 3 recommendations for a specific model and condition.

    Args:
        model_type: Embedding model type (csv, cosmetic, scibert, biobert, chembert)
        condition: Skin condition to analyze
        method: Dimensionality reduction method
        products_path: Path to products CSV file
        conditions_path: Path to conditions CSV file

    Returns:
        Dictionary with recommendation data or None if error
    """
    import unified_recommendation

    # Set the embedding model
    original_model = unified_recommendation.EMBEDDING_MODEL
    unified_recommendation.EMBEDDING_MODEL = model_type

    try:
        print(f"  Loading {model_type.upper()} model for {condition}...")

        # Initialize engine
        from unified_recommendation import UnifiedSkincareRecommendationEngine
        engine = UnifiedSkincareRecommendationEngine(
            products_path=products_path,
            conditions_path=conditions_path
        )

        # Get top 3 recommendations
        recommendations = engine.recommend(condition, top_n=3)

        # Collect embeddings
        embeddings = []
        labels = []
        colors = []
        sizes = []

        # Beneficial and harmful ingredients
        ben_embedding = engine.condition_embeddings[condition]
        harm_embedding = engine.harmful_embeddings[condition]
        embeddings.extend([ben_embedding, harm_embedding])
        colors.extend(['green', 'red'])
        sizes.extend([400, 400])
        labels.extend(['Beneficial', 'Harmful'])

        # Top 3 products
        for i, rec in enumerate(recommendations, 1):
            product_embedding = engine.product_embeddings[rec['index']]
            embeddings.append(product_embedding)

            if i == 1:
                colors.append('gold')
                sizes.append(350)
            elif i == 2:
                colors.append('orange')
                sizes.append(300)
            else:
                colors.append('lightsalmon')
                sizes.append(250)

            labels.append(f"#{i}")

        embeddings = np.array(embeddings)

        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(4, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(embeddings)-1))
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(4, len(embeddings)-1))
                embeddings_2d = reducer.fit_transform(embeddings)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 9))

        # Plot points
        for i in range(len(embeddings_2d)):
            ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                      c=[colors[i]], s=sizes[i], alpha=0.7,
                      edgecolors='black', linewidths=2.5, zorder=10)

            # Add labels
            ax.annotate(labels[i],
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=11, fontweight='bold', ha='center', va='center',
                       zorder=20)

        # Title with scores
        title = f'{model_type.upper()} - Top 3 Recommendations for {condition}\n'
        title += f'Scores: #{1}: {recommendations[0]["score"]:.3f} | '
        title += f'#{2}: {recommendations[1]["score"]:.3f} | '
        title += f'#{3}: {recommendations[2]["score"]:.3f}'

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                       markersize=12, markeredgecolor='black', markeredgewidth=2,
                       label='Beneficial Ingredients'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, markeredgecolor='black', markeredgewidth=2,
                       label='Harmful Ingredients'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                       markersize=11, markeredgecolor='black', markeredgewidth=2,
                       label='Rank #1'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                       markersize=10, markeredgecolor='black', markeredgewidth=2,
                       label='Rank #2'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightsalmon',
                       markersize=9, markeredgecolor='black', markeredgewidth=2,
                       label='Rank #3')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 framealpha=0.9, edgecolor='black')

        # Add recommendation details as text
        text_str = "Top Recommendations:\n"
        for rec in recommendations:
            text_str += f"#{rec['rank']}: {rec['brand']} - {rec['name'][:35]}...\n"

        ax.text(0.02, 0.02, text_str, transform=ax.transAxes,
               fontsize=8, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save plot with descriptive name
        safe_condition = condition.replace(' ', '_').replace('/', '_')
        filename = f'{model_type}_{safe_condition}_{method}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {filename}")

        plt.close()

        # Restore original model
        unified_recommendation.EMBEDDING_MODEL = original_model

        return {
            'model': model_type,
            'condition': condition,
            'recommendations': recommendations,
            'filename': filename
        }

    except Exception as e:
        print(f"    ✗ Error with {model_type} for {condition}: {e}")
        unified_recommendation.EMBEDDING_MODEL = original_model
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations for all conditions and all embedding models'
    )
    parser.add_argument('--method', type=str, choices=['tsne', 'umap'],
                       default='tsne', help='Dimensionality reduction method')
    parser.add_argument('--models', nargs='+',
                       choices=['csv', 'cosmetic', 'scibert', 'biobert', 'chembert'],
                       default=['csv', 'cosmetic', 'scibert', 'biobert', 'chembert'],
                       help='Embedding models to use')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Store original directory and absolute paths
    original_dir = os.getcwd()
    products_path = os.path.join(original_dir, 'cosmetic_p.csv')
    conditions_path = os.path.join(original_dir, 'dermatology_ingredients.csv')

    # Change to output directory
    os.chdir(args.output_dir)

    print("\n" + "="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)
    print(f"Method: {args.method.upper()}")
    print(f"Models: {', '.join([m.upper() for m in args.models])}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80 + "\n")

    # Get list of conditions
    conditions_df = pd.read_csv(conditions_path)
    conditions = list(conditions_df['Condition'].unique())

    print(f"Found {len(conditions)} conditions: {', '.join(conditions)}\n")

    # Track results
    results = []
    total = len(conditions) * len(args.models)
    current = 0

    # Generate plots for each combination
    for condition in conditions:
        print(f"\nCondition: {condition}")
        print("-" * 80)

        for model_type in args.models:
            current += 1
            print(f"[{current}/{total}] Processing {model_type.upper()}...")

            result = plot_top3_for_model(model_type, condition, args.method,
                                        products_path, conditions_path)
            if result:
                results.append(result)

    # Change back to original directory
    os.chdir(original_dir)

    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    print(f"Total plots generated: {len(results)}/{total}")
    print(f"Output directory: {args.output_dir}")

    # Create a CSV summary
    summary_data = []
    for result in results:
        for rec in result['recommendations']:
            summary_data.append({
                'model': result['model'],
                'condition': result['condition'],
                'rank': rec['rank'],
                'brand': rec['brand'],
                'name': rec['name'],
                'score': rec['score'],
                'beneficial_similarity': rec['beneficial_similarity'],
                'harmful_similarity': rec['harmful_similarity'],
                'price': rec['price'],
                'category': rec['category'],
                'filename': result['filename']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'recommendations_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS BY MODEL")
    print("="*80)

    for model in args.models:
        model_data = summary_df[summary_df['model'] == model]
        if len(model_data) > 0:
            avg_score = model_data[model_data['rank'] == 1]['score'].mean()
            print(f"{model.upper():12} - Avg Top-1 Score: {avg_score:.4f}")

    print("\n" + "="*80)
    print("STATISTICS BY CONDITION")
    print("="*80)

    for condition in conditions:
        cond_data = summary_df[summary_df['condition'] == condition]
        if len(cond_data) > 0:
            best_model = cond_data[cond_data['rank'] == 1].nlargest(1, 'score').iloc[0]
            print(f"{condition:20} - Best Model: {best_model['model'].upper():10} "
                  f"(Score: {best_model['score']:.4f})")

    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"All plots saved in: {args.output_dir}/")
    print(f"Total files: {len(results)} PNG files + 1 CSV summary")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
