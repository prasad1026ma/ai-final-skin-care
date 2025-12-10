"""
Clean up ingredient lists by removing specific filler words.
"""
import pandas as pd

def clean_filler_words(text):
    """
    Remove specific filler words from ingredient text.
    """
    if pd.isna(text) or not text:
        return text

    # List of filler words to remove
    filler_words = ['that', 'with', 'and', 'or', 'in', 'the']

    # Replace each filler word (case-insensitive)
    for word in filler_words:
        # Remove the word with spaces around it
        text = text.replace(f' {word} ', ' ')
        text = text.replace(f' {word.capitalize()} ', ' ')

    # Clean up extra spaces
    text = ' '.join(text.split())

    return text


def main():
    # Read the CSV
    df = pd.read_csv('dermatology_ingredients.csv')

    print("="*70)
    print("REMOVING FILLER WORDS: that, with, and, or, in, the")
    print("="*70)

    # Clean each row
    for idx, row in df.iterrows():
        condition = row['Condition']
        print(f"\n{condition}:")

        # Clean harmful ingredients
        original_harmful = row['Ingredients_to_Avoid']
        cleaned_harmful = clean_filler_words(original_harmful)
        if original_harmful != cleaned_harmful:
            print(f"  Avoid - Before: {original_harmful}")
            print(f"  Avoid - After:  {cleaned_harmful}")

        # Clean beneficial ingredients
        original_beneficial = row['Ingredients_to_Use']
        cleaned_beneficial = clean_filler_words(original_beneficial)
        if original_beneficial != cleaned_beneficial:
            print(f"  Use - Before: {original_beneficial}")
            print(f"  Use - After:  {cleaned_beneficial}")

        # Update dataframe
        df.at[idx, 'Ingredients_to_Avoid'] = cleaned_harmful
        df.at[idx, 'Ingredients_to_Use'] = cleaned_beneficial

    # Save cleaned version
    df.to_csv('dermatology_ingredients.csv', index=False)
    print(f"\n{'='*70}")
    print("✓ Updated dermatology_ingredients.csv")
    print("="*70)


if __name__ == "__main__":
    main()
