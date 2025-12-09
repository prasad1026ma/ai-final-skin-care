import pandas as pd
import numpy as np
import time
import sys
from Reccomendation.recommendation import SkincareRecommendationEngine
from Classification.classification_pipeline import run_classification_pipeline


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    print("\n" + "=" * 70)
    print(Colors.BOLD + Colors.CYAN + """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              SKIN LESION ANALYSIS & TREATMENT FINDER          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """ + Colors.ENDC)
    print("=" * 70 + "\n")


def print_loading_bar(duration=2):
    print(Colors.CYAN + "Analyzing image: " + Colors.ENDC)
    bar_length = 40
    for i in range(bar_length + 1):
        percent = (i / bar_length) * 100
        filled = '█' * i
        empty = '░' * (bar_length - i)
        sys.stdout.write(f'\r[{filled}{empty}] {percent:.0f}%')
        sys.stdout.flush()
        time.sleep(duration / bar_length)
    print("\n")


def print_diagnosis_box(predicted_class, confidence):
    """Print diagnosis results in a nice box"""
    print("\n" + Colors.BOLD + "┌" + "─" * 68 + "┐" + Colors.ENDC)
    print(Colors.BOLD + "│" + " " * 68 + "│" + Colors.ENDC)

    # Condition name
    condition_text = f"DETECTED CONDITION: {predicted_class.upper()}"
    padding = (68 - len(condition_text)) // 2
    print(Colors.BOLD + "│" + " " * padding + Colors.GREEN + condition_text + Colors.ENDC +
          " " * (68 - padding - len(condition_text)) + Colors.BOLD + "│" + Colors.ENDC)

    # Confidence with color coding
    if confidence >= 70:
        conf_color = Colors.GREEN
    elif confidence >= 50:
        conf_color = Colors.YELLOW
    else:
        conf_color = Colors.RED

    conf_text = f"Confidence: {confidence:.1f}%"
    padding = (68 - len(conf_text) + len(conf_color) + len(Colors.ENDC)) // 2
    print(Colors.BOLD + "│" + " " * padding + conf_color + conf_text + Colors.ENDC +
          " " * (68 - padding - len(conf_text)) + Colors.BOLD + "│" + Colors.ENDC)

    print(Colors.BOLD + "│" + " " * 68 + "│" + Colors.ENDC)
    print(Colors.BOLD + "└" + "─" * 68 + "┘" + Colors.ENDC + "\n")


def print_confidence_bar(confidence):
    """Visual confidence bar"""
    bar_length = 30
    filled = int((confidence / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)

    if confidence >= 70:
        color = Colors.GREEN
    elif confidence >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    print(f"   {color}{bar}{Colors.ENDC} {confidence:.1f}%")

def print_star_rating(rating):
    """Convert numeric rating to star display"""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star

    stars = "⭐" * full_stars + ("✨" if half_star else "") + "☆" * empty_stars
    return stars
def print_recommendation(recommendations):
    """Print recommendations with beautiful formatting"""
    if not recommendations:
        print(Colors.RED + "  No recommendations found for this condition." + Colors.ENDC)
        return

    print(Colors.BOLD + Colors.BLUE + "           RECOMMENDED TREATMENT PRODUCTS           " + Colors.ENDC)

    for i, rec in enumerate(recommendations, 1):
        # Rank badge
        rank_colors = [Colors.YELLOW, Colors.CYAN, Colors.GREEN]
        rank_color = rank_colors[i - 1] if i <= 3 else Colors.ENDC
        rank_badge = f"#{i} BEST MATCH" if i == 1 else f"#{i} RECOMMENDED"

        print(rank_color + Colors.BOLD + rank_badge + Colors.ENDC)
        print("─" * 70)

        # Product name
        product_name = f"{rec['brand']} - {rec['name']}"
        print(Colors.CYAN + Colors.BOLD + product_name + Colors.ENDC)

        # Price and Rating
        stars = print_star_rating(rec['rating'])
        price_text = f"${rec['price']:.2f}"
        rating_text = f"{stars} {rec['rating']:.1f}/5.0"

        print(Colors.GREEN + price_text + Colors.ENDC + "  |  " + rating_text)

        print()  # Empty line

        # Scores section
        print(Colors.UNDERLINE + "Match Scores:" + Colors.ENDC)

        # Overall score
        overall_pct = rec['score'] * 100
        bar_length = 30
        filled = int((overall_pct / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(Colors.YELLOW + f"  Overall Match:  {bar} {overall_pct:5.1f}%" + Colors.ENDC)

        # Beneficial similarity
        ben_pct = rec['beneficial_similarity'] * 100
        filled = int((ben_pct / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(Colors.GREEN + f"  Beneficial:     {bar} {ben_pct:5.1f}%" + Colors.ENDC)

        # Safety
        harm_pct = rec['harmful_similarity'] * 100
        safe_pct = 100 - harm_pct
        filled = int((safe_pct / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(Colors.CYAN + f"  Harmful:         {bar} {safe_pct:5.1f}%" + Colors.ENDC)

        print("\n")  # Space between products

def run_troubleshooter():
    """Main troubleshooter function with improved UI"""
    print_banner()

    # Get image path
    print(Colors.BOLD + "Please provide the skin lesion image:" + Colors.ENDC)
    image_path = input(Colors.CYAN + "   Path: " + Colors.ENDC).strip()

    if not image_path:
        print(Colors.RED + "Error: No image path provided!" + Colors.ENDC)
        return

    # Loading animation
    print_loading_bar(duration=1.5)

    try:
        # Run classification
        print(Colors.CYAN + "Running Image Classification..." + Colors.ENDC)
        predicted_class, confidence = run_classification_pipeline(image_path, 224)

        # Display results
        print_diagnosis_box(predicted_class, confidence)

        # Initialize recommendation engine
        print(Colors.CYAN + "Searching treatment database..." + Colors.ENDC)
        engine = SkincareRecommendationEngine(
            products_path='Reccomendation/cosmetic_p.csv',
            conditions_path='Reccomendation/dermatology_ingredients.csv'
        )

        # Get recommendations
        recommendations = engine.recommend_top_3(predicted_class)
        print_recommendation(recommendations)

        print(Colors.BOLD + Colors.GREEN + "Analysis complete! Thank You!" + Colors.ENDC)

    except FileNotFoundError:
        print(Colors.RED + f"Error: Image file not found at '{image_path}'" + Colors.ENDC)
    except ValueError as e:
        print(Colors.RED + f"Error: {e}" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"Unexpected error: {e}" + Colors.ENDC)


if __name__ == "__main__":
    try:
        run_troubleshooter()
    except KeyboardInterrupt:
        print(Colors.YELLOW + "Analysis cancelled. Goodbye!" + Colors.ENDC)
    except Exception as e:
        print(Colors.RED + f"Fatal error: {e}" + Colors.ENDC)



