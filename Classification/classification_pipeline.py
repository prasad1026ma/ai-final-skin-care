import torch
import torch.nn as nn
from PIL import Image
from Classification.modeling.res_net import build_resnet
from Classification.skin_dataset import SkinDataset
import pandas as pd

def load_model(model_path='best_model.pth', num_classes=5, device=None):
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = build_resnet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

def load_label_mapping(mapping_file='label_mapping.csv'):
    df = pd.read_csv(mapping_file)
    return {row['label_idx']: row['condition_name'] for _, row in df.iterrows()}

def predict(image_path, model, device, label_mapping, input_size=224):
    img = Image.open(image_path).convert('RGB')
    transform = SkinDataset.get_transforms(train=False, input_size=input_size)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)

    predicted_label = label_mapping[top_idx.item()]
    confidence = top_prob.item() * 100
    return predicted_label, confidence


def run_classification_pipeline(image_path, input_size):
    model_path = 'Classification/modeling/best_model.pth'
    mapping_file = 'Classification/data/label_mapping.csv'
    model, device = load_model(model_path=model_path, num_classes=5)
    label_mapping = load_label_mapping(mapping_file)

    # Run prediction
    predicted_class, confidence = predict(image_path, model, device, label_mapping, input_size)

    return predicted_class, confidence

if __name__ == "__main__":
    image_path = input("Enter the path to the skin lesion image: ").strip()
    input_size = 224

    predicted_class, confidence = run_classification_pipeline(image_path, input_size)
    print(f"\nPredicted Skin Lesion: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
