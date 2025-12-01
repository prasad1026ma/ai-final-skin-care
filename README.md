# ai-final-skin-care

## Classification Pipeline

### Data Cleaning
- Remove unreadable or corrupted image files 
- Resize all images to a consistent input size (e.g., 224×224)
- Normalize pixel intensities 
- Apply augmentation (rotation, flip, color jitter) to improve generalization 
- Ensure every image is paired with a correct diagnostic label

### Model Training
- Train a deep convolutional neural network on cleaned and augmented images 
- Use batch normalization + dropout to stabilize training and reduce overfitting 
- Optimize using cross-entropy loss and the Adam optimizer
- Save the checkpoint with the best performance
### Model Analysis
- Evaluate final model using accuracy, precision, recall, and F1 score per class 
- Create a confusion matrix to highlight misclassified conditions
- Analyze per class performance to find where discrepencies occur

## Recommendation Algorithm


### Data Cleaning


### Model Training

### Model Analysis