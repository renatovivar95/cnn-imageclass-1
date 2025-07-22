## Image Classification with CNN

## Crater Detection Dataset Description

For this task, the label is either `0` (without crater) or `1` (with crater). The images are 48x48 matrices in grayscale.

In grayscale image data, pixel values range from 0 to 255 (as in this case), where:
- `0` represents black,
- `255` represents white,
- intermediate values represent various shades of gray.

As a normalization step, the data was divided by 255, scaling pixel values to a range between 0 and 1. This is important for several reasons:
- Improving model performance,
- Reducing computational complexity,
- Enhancing generalization.

### Class Imbalance

The given dataset is imbalanced:
- **63.85%** of images are labeled as `1` (with crater),
- **36.15%** are labeled as `0` (without crater).

Although the dataset is imbalanced, this case is not critical and can be addressed using appropriate techniques.

### Additional Data

In addition to the labeled training set, an extra dataset without labels was also provided.

Figures below show examples of the dataset:
- **Figure 1**: Example of an image with label `1` (with crater),
- **Figure 2**: Example of an image with label `0` (without crater).

*(Refer to `Figure 1` and `Figure 2` as shown in the corresponding documentation or notebook.)*


# CNN Binary Image Classification with Semi-Supervised Learning

This project trains a Convolutional Neural Network (CNN) to classify 48x48 grayscale images using binary labels (0 or 1). It also implements a basic semi-supervised learning approach by leveraging an additional unlabeled dataset.

---

## 📁 Project Structure

├── cnn_classifier.py # Main Python script
├── DATA/
│ ├── Xtrain1.npy # Main training image data (flattened)
│ ├── Ytrain1.npy # Corresponding binary labels
│ ├── Xtrain1_extra.npy # Extra unlabeled image data
│ └── Xtest1.npy # Test image data for final prediction
├── PLOTS/
│ ├── learning_curve.png # Saved loss plot
│ └── Precision-Recall_Curve.png # Saved PR curve
└── DATA/
└── y_pred_final.npy # Output predictions for submission



---

## 🚀 What the Script Does

The main script performs the following:

1. **Loads Training and Extra Data**
   - Reads labeled and extra unlabeled images from `.npy` files.
   - Reshapes and normalizes them for CNN input.

2. **First Training Phase**
   - Trains a CNN on labeled training data.
   - Evaluates on a held-out validation set.
   - Saves loss and precision-recall curves.

3. **Pseudo-labeling Phase**
   - Uses the first model to predict on the unlabeled extra dataset.
   - Selects the samples confidently predicted as class 0 (threshold: 0.25).
   - Labels them as `0` and appends them to the original training set.

4. **Second Training Phase**
   - Trains a new CNN model on the augmented dataset.
   - Evaluates performance and saves the model and plots.

5. **Final Prediction**
   - Uses the second model to predict labels for the final test set (`Xtest1.npy`).
   - Saves predictions to `DATA/y_pred_final.npy`.

---

## 🧠 CNN Architecture

- `Conv2D(16)` → `MaxPooling2D`
- `Conv2D(32)` → `MaxPooling2D`
- `Flatten` → `Dense(8)` → `Dropout(0.5)`
- `Dense(1, sigmoid)` for binary classification

---

## 🧪 Evaluation Metrics

- Accuracy
- F1 Score
- Precision
- Recall
- Precision-Recall Curve (saved to `PLOTS/`)

---

## 💻 How to Run the Script

### 1. Ensure Python Dependencies Are Installed

```bash
pip install numpy matplotlib scikit-learn tensorflow
```

```bash
python cnn_script.py
```

