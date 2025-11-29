#  Crop Classification with ResNet-50 and Grad-CAM

## Project Summary
This project builds a deep learning-based classifier to identify agricultural crops from images. A fine-tuned ResNet-50 model is used as the core classifier, with Grad-CAM visual explanations and an interactive Gradio demo for real-time predictions.


https://github.com/user-attachments/assets/6ea5350a-51e5-4cc6-bbfc-e5d9532513eb



## Model Architecture
- **Base model**: ResNet-50 (pretrained on ImageNet)
- **Fine-tuned layers**: `layer3`, `layer4`, and final `fc` layer
- **Input size**: 224×224 pixels (RGB)
- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (`lr=0.001`)
- **Epochs trained**: 15
- **Batch size**: 4

##  Dataset
- **Source**: Folder of labeled images (structured by class folder)
- **Classes included**: `banana`, `cotton`, `cherry`, `cardamom`, `clove`, etc.
- **Split**: 80% training, 20% validation using PyTorch `random_split`

## Results
- **Validation Accuracy**: **86.06%**
- **Confusion Matrix**: Visualized using `sklearn`
- **Grad-CAM**: Highlighted model attention over discriminative regions
- **CSV Export**: True vs Predicted labels stored in `crop_predictions.csv`

## Gradio Interface
An interactive Gradio web app allows:
- Uploading a crop image
- Getting the model’s prediction
- Viewing the **Grad-CAM overlay** to interpret the decision

```python
gr.Interface(fn=classify_and_explain, inputs=gr.Image(...), outputs=[...]).launch()
```

## Advanced Features
- `generate_gradcam()` for interpretability
- Confidence scores and superimposed heatmaps
- Batch prediction via notebook and CSV

## Future Improvements
- Add top-3 prediction display
- Deploy Gradio app to Hugging Face Spaces or Streamlit Cloud
- Train with more diverse crop types and lighting conditions
- Export model with ONNX for mobile inference
