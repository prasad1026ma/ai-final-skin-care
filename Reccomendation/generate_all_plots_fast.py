"""
Fast generation of all visualizations - loads each model only once.

This optimized script loads each embedding model once and processes all conditions,
instead of reloading models for each condition.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse


def plot_and_save(embeddings_2d, labels, colors, sizes, model_type, condition,
                  recommendations, method, output_dir):
    """Create and save plot."""

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot points
    for i in range(len(embeddings_2d)):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                  c=[colors[i]], s=sizes[i], alpha=0.7,
                  edgecolors='black', linewidths=2.5, zorder=10)
        ax.annotate(labels[i],
                   (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   fontsize=11, fontweight='bold', ha='center', va='center',
                   zorder=20)

    # Title
    title = f'{model_type.upper()} - Top 3 for {condition}\n'
    title += f'Scores: #{1}: {recommendations[0]["score"]:.3f} | '
    title += f'#{2}: {recommendations[1]["score"]:.3f} | '
    title += f'#{3}: {recommendations[2]["score"]:.3f}'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=12, markeredgecolor='black', markeredgewidth=2,
                   label='Beneficial'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=12, markeredgecolor='black', markeredgewidth=2,
                   label='Harmful'),
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

    # Add text
    text_str = "Top Recommendations:\n"
    for rec in recommendations:
        text_str += f"#{rec['rank']}: {rec['brand']} - {rec['name'][:35]}...\n"

    ax.text(0.02, 0.02, text_str, transform=ax.transAxes,
           fontsize=8, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    safe_condition = condition.replace(' ', '_').replace('/', '_')
    filename = f'{model_type}_{safe_condition}_{method}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def process_model(model_type, conditions, method, output_dir,
                  products_path, conditions_path):
    """Process all conditions for a single model."""

    import unified_recommendation

    print(f"\n{'='*80}")
    print(f"Processing {model_type.upper()} Model")
    print(f"{'='*80}")

    # Set embedding model
    unified_recommendation.EMBEDDING_MODEL = model_type

    # Load engine ONCE
    print(f"Loading {model_type.upper()} model...")
    from unified_recommendation import UnifiedSkincareRecommendationEngine
    engine = UnifiedSkincareRecommendationEngine(
        products_path=products_path,
        conditions_path=conditions_path
    )
    print(f"✓ Model loaded\n")

    results = []

    for i, condition in enumerate(conditions, 1):
        try:
            print(f"[{i}/{len(conditions)}] {condition}...", end=' ')

            # Get recommendations
            recommendations = engine.recommend(condition, top_n=3)

            # Collect embeddings
            embeddings = []
            labels = []
            colors = []
            sizes = []

            # Beneficial and harmful
            ben_embedding = engine.condition_embeddings[condition]
            harm_embedding = engine.harmful_embeddings[condition]
            embeddings.extend([ben_embedding, harm_embedding])
            colors.extend(['green', 'red'])
            sizes.extend([400, 400])
            labels.extend(['Beneficial', 'Harmful'])

            # Top 3 products
            for j, rec in enumerate(recommendations, 1):
                product_embedding = engine.product_embeddings[rec['index']]
                embeddings.append(product_embedding)
                if j == 1:
                    colors.append('gold')
                    sizes.append(350)
                elif j == 2:
                    colors.append('orange')
                    sizes.append(300)
                else:
                    colors.append('lightsalmon')
                    sizes.append(250)
                labels.append(f"#{j}")

            embeddings = np.array(embeddings)

            # Dimensionality reduction
            if method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42,
                             perplexity=min(4, len(embeddings)-1))
                embeddings_2d = reducer.fit_transform(embeddings)
            elif method == 'umap':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42,
                                      n_neighbors=min(5, len(embeddings)-1))
                    embeddings_2d = reducer.fit_transform(embeddings)
                except ImportError:
                    reducer = TSNE(n_components=2, random_state=42,
                                 perplexity=min(4, len(embeddings)-1))
                    embeddings_2d = reducer.fit_transform(embeddings)

            # Plot and save
            filename = plot_and_save(embeddings_2d, labels, colors, sizes,
                                   model_type, condition, recommendations,
                                   method, output_dir)

            print(f"✓ Saved: {filename}")

            # Store results
            results.append({
                'model': model_type,
                'condition': condition,
                'recommendations': recommendations,
                'filename': filename
            })

        except Exception as e:
            print(f"✗ Error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fast generation of all visualizations'
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

    # Get absolute paths
    products_path = os.path.abspath('cosmetic_p.csv')
    conditions_path = os.path.abspath('dermatology_ingredients.csv')

    print("\n" + "="*80)
    print("FAST PLOT GENERATION - All Conditions × All Models")
    print("="*80)
    print(f"Method: {args.method.upper()}")
    print(f"Models: {', '.join([m.upper() for m in args.models])}")
    print(f"Output: {args.output_dir}/")
    print("="*80)

    # Get conditions
    conditions_df = pd.read_csv(conditions_path)
    conditions = list(conditions_df['Condition'].unique())
    print(f"\nConditions ({len(conditions)}): {', '.join(conditions)}")

    # Process each model
    all_results = []
    for model_type in args.models:
        results = process_model(model_type, conditions, args.method,
                              args.output_dir, products_path, conditions_path)
        all_results.extend(results)

    # Create summary CSV
    summary_data = []
    for result in all_results:
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

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total plots: {len(all_results)}")
    print(f"Output dir: {args.output_dir}/")
    print(f"Summary CSV: {summary_path}")

    print("\n" + "="*80)
    print("BEST MODEL PER CONDITION (by top-1 score)")
    print("="*80)
    for condition in conditions:
        cond_data = summary_df[summary_df['condition'] == condition]
        if len(cond_data) > 0:
            best = cond_data[cond_data['rank'] == 1].nlargest(1, 'score').iloc[0]
            print(f"{condition:30} | {best['model'].upper():10} | Score: {best['score']:.4f}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
