# AI has been used to make the plots verbose and pretty. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import argparse

from unified_recommendation import UnifiedSkincareRecommendationEngine, EMBEDDING_MODEL


def plot_top3_unified(engine, condition: str, method: str = 'tsne', save_plot: bool = True):
    """
    Plot embeddings for top 3 recommendations from unified engine.

    Args:
        engine: Unified recommendation engine instance
        condition: Skin condition to analyze
        method: Dimensionality reduction method ('tsne' or 'umap')
        save_plot: Whether to save the plot
    """
    print(f"\n{'='*70}")
    print(f"Top 3 Recommendations for {condition}")
    print(f"Using {engine.model_type.upper()} embeddings")
    print(f"{'='*70}")

    # Get top 3 recommendations
    recommendations = engine.recommend(condition, top_n=3)

    # Collect embeddings
    embeddings = []
    labels = []
    colors = []
    sizes = []

    # Add beneficial ingredient embedding
    ben_embedding = engine.condition_embeddings[condition]
    embeddings.append(ben_embedding)
    labels.append(f"Beneficial Ingredients")
    colors.append('green')
    sizes.append(400)

    # Add harmful ingredient embedding
    harm_embedding = engine.harmful_embeddings[condition]
    embeddings.append(harm_embedding)
    labels.append(f"Harmful Ingredients")
    colors.append('red')
    sizes.append(400)

    # Add top 3 product embeddings
    for i, rec in enumerate(recommendations, 1):
        product_embedding = engine.product_embeddings[rec['index']]
        embeddings.append(product_embedding)

        product_label = f"#{i}: {rec['brand'][:15]}"
        labels.append(product_label)

        # Color gradient based on rank
        if i == 1:
            colors.append('gold')
            sizes.append(350)
        elif i == 2:
            colors.append('orange')
            sizes.append(300)
        else:
            colors.append('lightsalmon')
            sizes.append(250)

    embeddings = np.array(embeddings)

    # Dimensionality reduction
    print(f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]}D to 2D using {method.upper()}...")

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(4, len(embeddings)-1))
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed. Falling back to t-SNE...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(4, len(embeddings)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each point with labels
    for i in range(len(embeddings_2d)):
        ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                  c=[colors[i]], s=sizes[i], alpha=0.7,
                  edgecolors='black', linewidths=2.5, zorder=10)

    # Add annotations
    for i in range(len(embeddings_2d)):
        # Different annotation styles for ingredients vs products
        if i < 2:  # Ingredients
            bbox_props = dict(boxstyle='round,pad=0.7', facecolor=colors[i],
                            alpha=0.4, edgecolor='black', linewidth=2)
            ax.annotate(labels[i],
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       xytext=(15, 15), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=bbox_props,
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        else:  # Products
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor=colors[i],
                            alpha=0.5, edgecolor='black', linewidth=1.5)
            rank = i - 1
            product_text = f"{labels[i]}\nScore: {recommendations[rank-1]['score']:.3f}"
            ax.annotate(product_text,
                       (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       xytext=(10, -25), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=bbox_props,
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # Title with model info
    title = f'Top 3 Recommendations for {condition}\n'
    title += f'Embedding Model: {engine.model_type.upper()} | '
    title += f'Scores: #{1}: {recommendations[0]["score"]:.3f}, '
    title += f'#{2}: {recommendations[1]["score"]:.3f}, '
    title += f'#{3}: {recommendations[2]["score"]:.3f}'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=14, markeredgecolor='black', markeredgewidth=2,
                   label='Beneficial Ingredients'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=14, markeredgecolor='black', markeredgewidth=2,
                   label='Harmful Ingredients'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                   markersize=13, markeredgecolor='black', markeredgewidth=2,
                   label='Rank #1'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=12, markeredgecolor='black', markeredgewidth=2,
                   label='Rank #2'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightsalmon',
                   markersize=11, markeredgecolor='black', markeredgewidth=2,
                   label='Rank #3')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11,
             framealpha=0.9, edgecolor='black')

    plt.tight_layout()

    if save_plot:
        save_path = f'unified_{engine.model_type}_top3_{condition.lower()}_{method}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")

    plt.show()

    # Print detailed recommendation info
    print("\n" + "-"*70)
    print("DETAILED RECOMMENDATIONS")
    print("-"*70)
    for rec in recommendations:
        print(f"\nRank {rec['rank']}: {rec['brand']} - {rec['name']}")
        print(f"  Overall Score:          {rec['score']:.4f}")
        print(f"  Beneficial Similarity:  {rec['beneficial_similarity']:.4f}")
        print(f"  Harmful Similarity:     {rec['harmful_similarity']:.4f}")
        print(f"  Price:                  ${rec['price']}")
        print(f"  Category:               {rec['category']}")
        print(f"  Rating:                 {rec['rating']}")


def plot_all_conditions(engine, method: str = 'tsne'):
    """
    Plot top 3 recommendations for all supported conditions.

    Args:
        engine: Unified recommendation engine instance
        method: Dimensionality reduction method
    """
    conditions = engine.supported_conditions

    print(f"\n{'#'*70}")
    print(f"Plotting all {len(conditions)} conditions")
    print(f"{'#'*70}")

    for i, condition in enumerate(conditions, 1):
        print(f"\n[{i}/{len(conditions)}] Processing {condition}...")
        plot_top3_unified(engine, condition, method, save_plot=True)


def compare_embedding_models(condition: str, method: str = 'tsne'):
    """
    Compare top 3 recommendations across different embedding models.

    Args:
        condition: Skin condition to analyze
        method: Dimensionality reduction method
    """
    from unified_recommendation import TRANSFORMER_MODEL_NAMES

    embedding_models = ['csv', 'cosmetic', 'scibert', 'biobert', 'chembert']

    print(f"\n{'#'*70}")
    print(f"Comparing Embedding Models for {condition}")
    print(f"{'#'*70}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    all_recommendations = {}

    for idx, model_type in enumerate(embedding_models):
        print(f"\n[{idx+1}/{len(embedding_models)}] Processing {model_type.upper()} model...")

        # Temporarily change the embedding model
        import unified_recommendation
        original_model = unified_recommendation.EMBEDDING_MODEL
        unified_recommendation.EMBEDDING_MODEL = model_type

        try:
            # Initialize engine with this embedding model
            engine = UnifiedSkincareRecommendationEngine(
                products_path='cosmetic_p.csv',
                conditions_path='dermatology_ingredients.csv'
            )

            # Get top 3 recommendations
            recommendations = engine.recommend(condition, top_n=3)
            all_recommendations[model_type] = recommendations

            # Collect embeddings
            embeddings = []
            colors = []
            sizes = []

            # Beneficial and harmful
            ben_embedding = engine.condition_embeddings[condition]
            harm_embedding = engine.harmful_embeddings[condition]
            embeddings.extend([ben_embedding, harm_embedding])
            colors.extend(['green', 'red'])
            sizes.extend([300, 300])

            # Top 3 products
            for i, rec in enumerate(recommendations, 1):
                product_embedding = engine.product_embeddings[rec['index']]
                embeddings.append(product_embedding)
                if i == 1:
                    colors.append('gold')
                    sizes.append(250)
                elif i == 2:
                    colors.append('orange')
                    sizes.append(220)
                else:
                    colors.append('lightsalmon')
                    sizes.append(190)

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

            # Plot
            ax = axes[idx]
            for i in range(len(embeddings_2d)):
                ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                          c=[colors[i]], s=sizes[i], alpha=0.7,
                          edgecolors='black', linewidths=2)

                # Add simple labels
                if i == 0:
                    label = 'B'
                elif i == 1:
                    label = 'H'
                else:
                    label = f'#{i-1}'

                ax.annotate(label,
                           (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                           fontsize=10, fontweight='bold', ha='center', va='center')

            ax.set_title(f'{model_type.upper()}\nTop Score: {recommendations[0]["score"]:.3f}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
            ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error with {model_type}: {e}")
            axes[idx].text(0.5, 0.5, f'Error: {str(e)[:30]}',
                          ha='center', va='center', transform=axes[idx].transAxes)

        # Restore original model
        unified_recommendation.EMBEDDING_MODEL = original_model

    # Hide the last subplot (we have 5 models, 6 subplots)
    axes[-1].axis('off')

    plt.suptitle(f'Embedding Model Comparison - Top 3 Recommendations for {condition}\n'
                f'B=Beneficial, H=Harmful, #1-3=Product Ranks',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = f'unified_comparison_{condition.lower()}_{method}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {save_path}")

    plt.show()

    # Print comparison table
    print("\n" + "="*100)
    print("EMBEDDING MODEL COMPARISON TABLE")
    print("="*100)

    for rank in range(1, 4):
        print(f"\n{'Rank ' + str(rank):-^100}")
        for model_type in embedding_models:
            if model_type in all_recommendations:
                rec = all_recommendations[model_type][rank-1]
                print(f"{model_type.upper():12} | Score: {rec['score']:.4f} | "
                      f"Ben: {rec['beneficial_similarity']:.4f} | "
                      f"Harm: {rec['harmful_similarity']:.4f} | "
                      f"{rec['brand']} - {rec['name'][:40]}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize top 3 recommendations from unified recommendation engine'
    )
    parser.add_argument('--condition', type=str, default='Eczema',
                       help='Skin condition to analyze (default: Eczema)')
    parser.add_argument('--method', type=str, choices=['tsne', 'umap'],
                       default='tsne', help='Dimensionality reduction method')
    parser.add_argument('--all-conditions', action='store_true',
                       help='Plot top 3 for all conditions')
    parser.add_argument('--compare-models', action='store_true',
                       help='Compare all embedding models side-by-side')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to disk')

    args = parser.parse_args()

    if args.compare_models:
        # Compare all embedding models
        compare_embedding_models(args.condition, args.method)
    else:
        # Load unified engine with current embedding model
        print(f"\nLoading Unified Recommendation Engine...")
        print(f"Embedding Model: {EMBEDDING_MODEL.upper()}")

        engine = UnifiedSkincareRecommendationEngine(
            products_path='cosmetic_p.csv',
            conditions_path='dermatology_ingredients.csv'
        )

        if args.all_conditions:
            # Plot all conditions
            plot_all_conditions(engine, args.method)
        else:
            # Plot single condition
            plot_top3_unified(engine, args.condition, args.method,
                            save_plot=not args.no_save)


if __name__ == "__main__":
    main()
