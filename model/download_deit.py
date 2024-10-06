from transformers import DeiTForImageClassification, DeiTFeatureExtractor
import torch

# Load the feature extractor and model
feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

# Save the model locally
model.save_pretrained("./deit-base-distilled-patch16-224")
feature_extractor.save_pretrained("./deit-base-distilled-patch16-224")

print("Model and feature extractor downloaded and saved locally.")
