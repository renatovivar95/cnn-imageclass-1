#!/usr/bin/env python3

# IMPORTING LIBRARIES
import argparse
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import precision_recall_curve
import os

# FUNCTION TO PLOT A SINGLE IMAGE
def plot_image_48x48(imageX, n):
    image = imageX[n]
    image_reshaped = image.reshape(48, 48)
    plt.imshow(image_reshaped, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    plt.show()

# FUNCTION TO RESHAPE THE X DATA
def reshape_image_data(data):
    x_images = data.reshape((data.shape[0], 48, 48))
    x_images = x_images / 255.0
    x_images = x_images.reshape((x_images.shape[0], 48, 48, 1))
    return x_images

def get_unique_filepath(base_path, extension=".png"):
    counter = 0
    base_name = base_path
    full_path = f"{base_name}{extension}"
    
    while os.path.exists(full_path):
        counter += 1
        full_path = f"{base_name}_{counter}{extension}"
    
    return full_path

# CONVOLUTIONAL NEURAL NETWORK CLASSIFICATION FUNCTION
def cnn_bin_classification(train_data_x, train_data_y, test_data_x, test_data_y, threshold, l_rate, n_epochs, file_name=None):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((3, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    
    adam = keras.optimizers.Adam(learning_rate=l_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(train_data_x, train_data_y, epochs=n_epochs,
                        validation_data=(test_data_x, test_data_y))
    
    model.save(str(file_name) if file_name else "cnn_model.h5")

    os.makedirs("PLOTS", exist_ok=True)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plot_path = get_unique_filepath("PLOTS/learning_curve")
    plt.savefig(plot_path)
    plt.close()
    
    loss, accuracy = model.evaluate(test_data_x, test_data_y)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    y_pred_aux = model.predict(test_data_x)
    y_pred = (y_pred_aux > threshold).astype("int32")

    precision, recall, _ = precision_recall_curve(test_data_y, y_pred_aux)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plot_path = get_unique_filepath("PLOTS/Precision-Recall_Curve.png")
    plt.savefig(plot_path)
    plt.close()


    f1 = f1_score(test_data_y, y_pred, average='macro')
    precision_score_val = precision_score(test_data_y, y_pred)
    recall_score_val = recall_score(test_data_y, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision_score_val:.4f}")
    print(f"Recall: {recall_score_val:.4f}")

    return model

def main():
    parser = argparse.ArgumentParser(description="CNN Binary Classification with Semi-Supervised Learning")

    parser.add_argument('--initial_lr', type=float, default=0.0003, help='Learning rate for the first model')
    parser.add_argument('--final_lr', type=float, default=0.0004, help='Learning rate for the second model')
    parser.add_argument('--initial_epochs', type=int, default=75, help='Epochs for the first model')
    parser.add_argument('--final_epochs', type=int, default=50, help='Epochs for the second model')
    parser.add_argument('--threshold_initial', type=float, default=0.5, help='Prediction threshold for first model evaluation')
    parser.add_argument('--threshold_final', type=float, default=0.5, help='Prediction threshold for final model')

    args = parser.parse_args()

    x_train_1 = np.load('DATA/Xtrain1.npy') 
    y_train_1 = np.load('DATA/Ytrain1.npy') 
    x_train_extra = np.load('DATA/Xtrain1_extra.npy') 

    reshape_x = reshape_image_data(x_train_1)

    train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(
        reshape_x, y_train_1, test_size=0.2, random_state=42, stratify=y_train_1)

    model_1 = cnn_bin_classification(train_x_1, train_y_1, test_x_1, test_y_1,
                                     threshold=args.threshold_initial,
                                     l_rate=args.initial_lr,
                                     n_epochs=args.initial_epochs,
                                     file_name="model_initial.h5")
    model_1.summary()


    # Reshape the extra training data
    x_images_extra = reshape_image_data(x_train_extra)

    # Predict using the first model
    y_probs_extra = model_1.predict(x_images_extra)


    indices_predicted_as_zero = np.where(y_probs_extra < 0.25)[0]
    X_new_predicted_as_zero = x_train_extra[indices_predicted_as_zero]
    Y_new_predicted_as_zero = np.zeros(len(indices_predicted_as_zero),)
    X_new_predicted_as_zero_reshape = reshape_image_data(X_new_predicted_as_zero)


    data_new_x = np.concatenate((reshape_x, X_new_predicted_as_zero_reshape), axis=0)
    data_new_y = np.concatenate((y_train_1, Y_new_predicted_as_zero), axis=0)
    data_new_y = data_new_y.astype(int)
     
    print(data_new_x.shape, data_new_y.shape)

    train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(
        data_new_x, data_new_y, test_size=0.2, random_state=42, stratify=data_new_y)

    model_2 = cnn_bin_classification(train_x_1, train_y_1, test_x_1, test_y_1,
                                     threshold=args.threshold_final,
                                     l_rate=args.final_lr,
                                     n_epochs=args.final_epochs,
                                     file_name="model_final.h5")
    model_2.summary()


    x_test_final = np.load('DATA/Xtest1.npy')
    x_test_final_reshape = reshape_image_data(x_test_final)
    y_pred_final = (model_2.predict(x_test_final_reshape) > args.threshold_final).astype("int32")
    y_pred_final = y_pred_final.reshape(len(y_pred_final),)
    np.save("DATA/y_pred_final.npy", y_pred_final)

if __name__ == "__main__":
    main()
