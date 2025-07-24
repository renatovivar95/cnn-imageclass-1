# CNN Binary Image Classification with Semi-Supervised Learning

This project performs binary classification on 48x48 grayscale images to detect the presence of craters (`1` = with crater, `0` = without crater) using a Convolutional Neural Network (CNN). It incorporates a **semi-supervised learning** approach by leveraging additional unlabeled data through **pseudo-labeling**.

---

## Crater Detection Dataset Description

For this task, the label is either `0` (without crater) or `1` (with crater). The images are 48x48 matrices in grayscale.

In grayscale image data, pixel values range from 0 to 255:
- `0` represents black,
- `255` represents white,
- intermediate values represent shades of gray.

As a normalization step, pixel values are divided by 255 to scale them to the [0, 1] range. This:
- Improves model performance,
- Reduces computational complexity,
- Enhances generalization.

### Class Imbalance

The dataset is slightly imbalanced:
- **63.85%** of images are labeled `1` (with crater),
- **36.15%** are labeled `0` (without crater).

Although imbalanced, this can be mitigated using techniques like pseudo-labeling.

### Additional Data

An extra unlabeled dataset is provided for semi-supervised learning.

Example images:
- **Figure 1**: Image with crater (`1`)
- **Figure 2**: Image without crater (`0`)

---

## Project Structure

├── cnn_classifier.py # Main Python script

├── DATA/

│ ├── Xtrain1.npy # Labeled training images

│ ├── Ytrain1.npy # Binary labels for training

│ ├── Xtrain1_extra.npy # Unlabeled extra images

│ ├── Xtest1.npy # Final test images

│ └── y_pred_final.npy # Output predictions

├── MODELS/
│ ├── model_initial.h5 # First trained model (supervised)

│ └── model_final.h5 # Final model (after pseudo-labeling)

├── PLOTS/

│ ├── learning_curve.png # Training/validation loss plot

│ ├── Precision-Recall_Curve.png # PR curve

│ └── confusion_matrix.png # Confusion matrix



---

## What the Script Does

The main steps are:

1. **Load and Normalize Data**
   - Reshape and scale grayscale images.

2. **Initial Training Phase**
   - CNN trained on labeled data only.
   - Model saved and evaluated on a validation set.

3. **Pseudo-Labeling Phase**
   - Predict on unlabeled data.
   - Select high-confidence samples (probability < 0.25) as class `0`.
   - Augment training dataset.

4. **Final Training Phase**
   - Train a new CNN on the augmented dataset.
   - Evaluate and save metrics.

5. **Final Prediction**
   - Predict labels for the final test dataset.
   - Save predictions to `DATA/y_pred_final.npy`.

---

## CNN Architecture

- `Conv2D(16)` → `MaxPooling2D(3x3)`
- `Conv2D(32)` → `MaxPooling2D(2x2)`
- `Flatten` → `Dense(8)` → `Dropout(0.5)`
- `Dense(1, sigmoid)` for binary classification

---

## Evaluation Metrics

- Accuracy
- F1 Score (macro)
- Precision
- Recall
- **Precision-Recall Curve**
- **Confusion Matrix**

All plots are saved in the `PLOTS/` folder.

---

## How to Run the Script

### 1. Install Python Dependencies

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow
```

### 2. Run the Script
```bash
python cnn_classifier.py
```

### 3. Optional Arguments

You can customize the learning rate, epochs, and prediction thresholds:

```bash
python cnn_classifier.py \
  --initial_lr 0.0003 \
  --final_lr 0.0004 \
  --initial_epochs 75 \
  --final_epochs 50 \
  --threshold_initial 0.5 \
  --threshold_final 0.5
```

## Notes

- The script uses ModelCheckpoint to save the best models based on validation loss.
- The semi-supervised approach improves robustness by including pseudo-labeled examples from unlabeled data.
- Outputs such as metrics and visualizations help monitor performance.


## Sample Visualization Function

Use the plot_image_48x48(data, n) helper function to view individual samples.
