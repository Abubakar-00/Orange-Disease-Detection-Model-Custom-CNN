# Orange Leaf Disease Detection Model using Custom CNN

![Orange Leaf Diseases](https://img.shields.io/badge/Deep%20Learning-Orange%20Disease-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![Python](https://img.shields.io/badge/Python-3.7+-blue)

<p align="center">
    <img src="https://github.com/user-attachments/assets/f1076535-ddd9-48e8-83d3-3830aacae6b0">
</p>

A deep learning model built with TensorFlow to detect various diseases in orange leaves using a custom Convolutional Neural Network (CNN) architecture based on LeNet. The model can identify three different conditions: Healthy leaves, Chlorosis, and Canker.

## Features
- Custom CNN architecture based on LeNet
- Real-time disease prediction capabilities
- Batch normalization for training stability
- Dropout layers to prevent overfitting
- L2 regularization for weight optimization
- Comprehensive metrics tracking including:
  - True Positives/Negatives
  - False Positives/Negatives
  - Precision/Recall
  - AUC-ROC

## Dataset Structure
The dataset contains orange leaf images categorized into three classes:
1. Healthy - Normal green leaves
2. Chlorosis - Yellowing of leaves
3. Canker - Spotted lesions on leaves

## Model Architecture
```python
- Input Layer (256x256x3)
- Convolutional Layer (6 filters, 3x3 kernel)
- Batch Normalization
- MaxPooling Layer (2x2)
- Dropout (0.5)
- Flatten Layer
- Dense Layer (100 units)
- Batch Normalization
- Dropout (0.5)
- Dense Layer (10 units)
- Batch Normalization
- Output Layer (3 units, softmax)
```

## Requirements
- TensorFlow 2.0+
- NumPy
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)
- IPython
- ipywidgets


## Usage
1. **Dataset Preparation**
```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle=True,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)
```

2. **Training the Model**
```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy', 'precision', 'recall', 'auc']
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    batch_size=32,
    epochs=50
)
```

3. **Making Predictions**
```python
# For single image prediction
predicted_class, confidence = predict_image_with_model(image_data, model)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")
```

## Training Details
- Image Size: 256x256 pixels
- Batch Size: 32
- Training Split: 80%
- Validation Split: 10%
- Test Split: 10%
- Epochs: 50
- Dropout Rate: 0.5
- L2 Regularization Rate: 0.01

## Model Evaluation Metrics
The model tracks comprehensive metrics including:
- Binary Accuracy
- Precision
- Recall
- AUC-ROC
- True Positives/Negatives
- False Positives/Negatives

## Training Visualization
The model includes loss visualization during training:
- Training loss vs. Validation loss plots
- Accuracy metrics tracking
- ROC curve analysis

## Key Model Features
1. **Preprocessing**
   - Image resizing to 256x256
   - Pixel value rescaling (1/255)
   - Batch normalization

2. **Regularization**
   - Dropout layers (50% rate)
   - L2 regularization (0.01)
   - Batch normalization

3. **Real-time Prediction**
   - Support for single image upload
   - Confidence score calculation
   - Interactive prediction interface

## Future Improvements
1. Implement data augmentation techniques
2. Add support for mobile deployment
3. Enhance the model with transfer learning
4. Add model interpretability features
5. Implement early stopping and learning rate scheduling

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
