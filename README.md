# Kidney Disease Classification Using Deep Learning on CT Images
This project focuses on classifying kidney diseases using CT images and multiple deep learning models: EfficientNetB0, MobileNetV2, ResNet50, InceptionV3, and Vision Transformer (ViT). The dataset used contains annotated CT images categorized into Normal, Cyst, Stone, and Tumor.

**Author**: Dheeraj Atul Salokhe  

---

## 🧠 Abstract

Kidney disease is a growing global health issue caused by factors like diabetes and hypertension. This project applies deep learning to classify kidney conditions—Normal, Cyst, Tumor, and Stone—using CT scan images. Five deep learning models (InceptionV3, MobileNetV2, EfficientNetB0, ResNet50, Vision Transformer) were evaluated. The best-performing model, **InceptionV3**, achieved **100% test accuracy**. Grad-CAM visualizations enhance model transparency.

---

## 📁 Dataset

- **Source**: [Kaggle – CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)  
- **Total Images**: 12,446  
  - Normal: 5,077  
  - Cyst: 3,709  
  - Tumor: 2,283  
  - Stone: 1,377  

---

## 🎯 Research Objectives

- Classify CT kidney images into four categories using deep learning.  
- Compare the performance of multiple architectures.  
- Visualize significant image regions with Grad-CAM to support interpretability.

---

## 🚀 Models Compared

| Model               | Precision | Recall | F1 Score | Accuracy | Test Accuracy | Test Loss |
|--------------------|-----------|--------|----------|----------|----------------|-----------|
| EfficientNetB0     | 0.1664    | 0.4079 | 0.2364   | 33.98%   | 40.59%         | 1.3407    |
| MobileNetV2        | 0.2525    | 0.2500 | 0.2350   | 33.98%   | 82.58%         | 0.6547    |
| ResNet50           | 0.2522    | 0.2504 | 0.2353   | 34.08%   | 99.82%         | 0.0046    |
| InceptionV3        | 0.2517    | 0.2490 | 0.2339   | 33.95%   | **100.00%**    | **0.0001**|
| Vision Transformer | **0.30**  | **0.33** | **0.31** | **31%** | 87.32%         | 0.3183    |

---

## 🧪 Evaluation Metrics

- Precision  
- Recall  
- F1 Score  
- Accuracy  
- Test Accuracy  
- Test Loss  
- Grad-CAM Heatmaps  

---

## 🔍 Grad-CAM for Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visually explain the decisions made by each deep learning model. This interpretability enhances trust among medical professionals by showing where the model focused its attention when predicting kidney diseases.

### 💡 Key Steps:

- **Gradient Backpropagation**: Calculated gradients of the predicted class with respect to the last convolutional layer.  
- **Feature Map Weighting**: Applied these gradients to the layer's activations to generate a coarse heatmap.  
- **Overlay Visualization**: Superimposed the heatmap onto the original CT image to highlight the most influential regions.  

### ✅ Insights:

- Grad-CAM heatmaps consistently focused on anatomically relevant regions associated with renal pathologies.  
- For example, cyst predictions aligned with visibly abnormal circular regions in the kidney.  
- Helped validate and fine-tune model performance through better preprocessing and training adjustments.  
- Enhances transparency and trust in AI-assisted diagnosis in healthcare.

---

## 🖥️ How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dheerajsalokhe/ct-scan-kidney-disease.git
   cd ct-scan-kidney-disease

Follow these steps to set up the project and run the classification models with Grad-CAM visualizations:

1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/ct-scan-kidney-disease.git
cd ct-scan-kidney-disease
🔁 Replace your-username with your actual GitHub username.

2. Install Required Libraries
Ensure you have Python 3.10 installed. Then install the dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy
Edit
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python pillow accelerate grad-cam kaggle jupyter
3. Download Dataset
Visit the Kaggle CT Kidney Dataset.

Download the dataset and extract it into your project directory.

Example folder path:

css
Copy
Edit
ct-scan-kidney-disease/
├── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/
│   ├── Normal/
│   ├── Cyst/
│   ├── Tumor/
│   └── Stone/
4. Launch Jupyter Notebook
Start the notebook interface:

bash
Copy
Edit
jupyter notebook
Then open:

text
Copy
Edit
Kidney_Disease_Classification.ipynb
and run the notebook cell-by-cell to:

Load and preprocess the dataset.

Train and evaluate models.

Generate Grad-CAM visualizations.

5. View Grad-CAM Outputs
At the end of the notebook, you’ll see Grad-CAM heatmaps overlaid on original CT images. These highlight regions the model focused on for classification.


🔍 Grad-CAM for Model Interpretability
To improve the trustworthiness and transparency of model predictions, Gradient-weighted Class Activation Mapping (Grad-CAM) was applied. This technique highlights the regions of the CT images that most influenced the model’s decisions.

📌 Implementation Highlights
Gradient Backpropagation: Computed gradients of the predicted class with respect to the final convolutional layer.

Feature Map Weighting: Used the gradients to weight the feature maps and generate a coarse heatmap.

Overlay Visualization: Superimposed the heatmap on the original CT image to visualize regions of interest.

📷 Visual Outputs
Grad-CAM outputs helped verify that the models were focusing on anatomically relevant areas for each condition:

Cyst: Heatmaps highlighted fluid-filled sac-like structures.

Tumor: Focused on abnormal growth regions.

Stone: Detected localized calcifications.

Normal: Uniform attention across renal structures.

These visual explanations enhance clinical trust and offer transparency in model decision-making, bridging the gap between AI systems and medical professionals.


