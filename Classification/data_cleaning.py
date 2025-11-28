import ast
import pandas as pd
from pathlib import Path
from Classification.constants import CONDITIONS
from Classification.skin_dataset import SkinDataset

def process_scin_dataset(cases_csv_path, labels_csv_path, images_base_dir, output_csv):
    cases_df = pd.read_csv(cases_csv_path, dtype={'case_id': str})
    labels_df = pd.read_csv(labels_csv_path, dtype={'case_id': str})
    merged_df = pd.merge(cases_df, labels_df, on='case_id', how='inner')
    label_column = 'weighted_skin_condition_label'

    merged_df = merged_df[merged_df[label_column].notna()].copy()

    unique_conditions = sorted(merged_df[label_column].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_conditions)}

    images_base = Path(images_base_dir)
    records = []

    image_path_cols = [col for col in merged_df.columns if 'image' in col.lower() and 'path' in col.lower()]

    if not image_path_cols:
        raise ValueError("No image path columns found in the dataset!")

    for _, row in merged_df.iterrows():
        case_id = row['case_id']
        condition = row[label_column]
        label_idx = label_to_idx[condition]

        # Process each image path column (image_1_path, image_2_path, image_3_path)
        for img_col in image_path_cols:
            if pd.notna(row[img_col]):
                img_path_str = row[img_col]
                img_filename = Path(img_path_str).name
                possible_paths = [
                    images_base / img_path_str,
                    images_base / img_filename,
                    images_base / 'dataset' / 'images' / img_filename,
                    images_base / 'images' / img_filename,
                ]

                img_full_path = None
                for path in possible_paths:
                    if path.exists():
                        img_full_path = path
                        break

                if img_full_path and img_full_path.exists():
                    # Determine image type from column name
                    if '1' in img_col:
                        img_type = 'image_1'
                    elif '2' in img_col:
                        img_type = 'image_2'
                    elif '3' in img_col:
                        img_type = 'image_3'
                    else:
                        img_type = 'unknown'

                    records.append({
                        'image_path': str(img_full_path),
                        'label': label_idx,
                        'condition_name': condition,
                        'case_id': case_id,
                        'image_type': img_type
                    })

    if len(records) == 0:
        print(f"Please check that images are located relative to: {images_base}")
        return None

    dataset_df = pd.DataFrame(records)
    dataset_df.to_csv(output_csv, index=False)

    return output_csv

def extract_top_condition(label_value):
    if pd.isna(label_value):
        return None
    if isinstance(label_value, str) and label_value.strip() in ['{}', '{ }', '']:
        return None
    # Check if it's a string representation of a dictionary
    if isinstance(label_value, str) and '{' in label_value and ':' in label_value:
        try:
            condition_dict = ast.literal_eval(label_value)

            if condition_dict:
                top_condition = max(condition_dict.items(), key=lambda x: x[1])
                return top_condition[0]  # Return the condition name
        except:
            # If parsing fails, return the original value
            return label_value
    else:
        # Already a simple string, return as-is
        return label_value


def load_dataset(csv_path='dataset.csv'):
    df = pd.read_csv(csv_path)

    df['labeled_condition_str'] = df['condition_name'].apply(extract_top_condition)
    df = df.dropna(subset=['labeled_condition_str', 'image_path']).copy()
    df = df[df['labeled_condition_str'].isin(CONDITIONS)].copy()

    condition_counts = df['labeled_condition_str'].value_counts()
    classes_to_remove = condition_counts[condition_counts < 2].index.tolist()

    if classes_to_remove:
        df = df[~df['labeled_condition_str'].isin(classes_to_remove)].copy()

    unique_labels = sorted(df['labeled_condition_str'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    df['labeled_condition'] = df['labeled_condition_str'].map(label_to_idx)

    mapping_df = pd.DataFrame([
        {'label_idx': idx, 'condition_name': name}
        for name, idx in label_to_idx.items()
    ])
    mapping_df.to_csv('label_mapping.csv', index=False)

    image_paths = df['image_path'].tolist()
    labels = df['labeled_condition'].tolist()
    return image_paths, labels


if __name__ == "__main__":
    try:
        dataset_path = process_scin_dataset(
            cases_csv_path='data/scin_cases.csv',
            labels_csv_path='data/scin_labels.csv',
            images_base_dir='data/images/',
            output_csv='data/dataset.csv'
        )
        image_paths,labels = load_dataset('data/dataset.csv')
        dataset = SkinDataset(
            image_paths,
            labels,
            transform=SkinDataset.get_transforms(train=True)
        )
    except Exception as e:
        print(f"Error: {e}")




