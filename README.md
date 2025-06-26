# Study-of-Deep-Learning-Models-for-Kidney-Disease-Classification-Using-CT-Images
Deep learning models (InceptionV3, MobileNetV2, EfficientNetB0, ResNet50 and ViT models) are used to classify kidney diseases (Normal, Cyst, Tumor, Stone) from CT images. Grad-CAM visualizations improve interpretability. InceptionV3 achieved 99.82% accuracy.
This project focuses on classifying kidney diseases using CT images and multiple deep learning models: EfficientNetB0, MobileNetV2, ResNet50, InceptionV3, and Vision Transformer (ViT). The dataset used contains annotated CT images categorized into Normal, Cyst, Stone, and Tumor.

📊 Model Performance Summary

Model

Precision

Recall

F1 Score

Accuracy

Test Accuracy

Test Loss

EfficientNetB0

0.1664

0.4079

0.2364

33.98%

40.59%

1.3407

MobileNetV2

0.2525

0.2500

0.2350

33.98%

82.58%

0.6547

ResNet50

0.2522

0.2504

0.2353

34.08%

99.82%

0.0046

InceptionV3

0.2517

0.2490

0.2339

33.95%

100.00%

0.0001

Vision Transformer (ViT)

0.30

0.33

0.31

31%

87.32%

0.3183

🧪 Libraries Used

Library

Version

Python

3.10.12

TensorFlow

2.13.0

Keras

2.13.1

NumPy

1.24.3

Pandas

2.0.3

Matplotlib

3.7.2

Scikit-learn

1.3.0

OpenCV-python

4.8.1

PIL (Pillow)

9.5.0

Accelerate

1.1.0

Grad-CAM

1.4.7

Kaggle

1.5.13

Jupyter

1.0.0

🔍 Grad-CAM for Model Interpretability

To enhance the interpretability of our deep learning models, Gradient-weighted Class Activation Mapping (Grad-CAM) was applied. Grad-CAM helps visualize which regions of CT kidney images influenced the model's predictions, providing crucial insights for medical experts.

🧠 Why Grad-CAM?

Grad-CAM helps bridge the gap between model decisions and clinical understanding by highlighting the anatomical regions contributing most to predictions. These visualizations help build trust with healthcare professionals by offering transparency in AI decision-making.

⚙️ Implementation Details:

Gradient Backpropagation: Gradients of the predicted class were computed with respect to the last convolutional layer.

Feature Map Weighting: These gradients were scaled and summed to generate a class-discriminative heatmap.

Overlay Visualization: The resulting heatmaps were overlaid on original CT images for an interpretable view of model focus.

🖼️ Insights:

Grad-CAM confirmed that the models correctly focused on anatomically relevant regions:

Cyst predictions aligned well with visible cyst-like structures.

Variations in Grad-CAM visualizations were used to guide preprocessing and improve model training.

These visual interpretations improve trust in model outputs and make AI applications in healthcare more explainable and applicable in clinical settings.
