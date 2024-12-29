# ASL to English And Other Indian Languages


## Contents
1. [Overview](#overview)
2. [File Descriptions](#file-descriptions)
   - [handPredict.py](#handpredictpy)
   - [aslVersion3.py](#aslversion3py)
   - [aslFinalMain.py](#aslfinalmainpy)
3. [Model Architecture](#model-architecture)
4. [How to Run the Project](#how-to-run-the-project)
5. [Training the Model](#training-the-model)
6. [Graphical Representation](#graphical-representation)
7. [Conclusion](#conclusion)

---

## Overview

This project is an American Sign Language (ASL) Translator that uses a pre-trained model to recognize and translate hand gestures into English and other South Indian languages. The application consists of three primary Python files: `handPredict.py`, `aslVersion3.py`, and `aslFinalMain.py`. The model has been trained using a Convolutional Neural Network (CNN) on a dataset of ASL alphabet images.

---

## File Descriptions

### handPredict.py

This file contains the logic for recognizing ASL letters. It loads the trained model, processes the input image, and predicts the corresponding ASL letter. It uses the following steps:
1. Pre-processes the image (resizing and normalizing).
2. Makes a prediction using the trained model.
3. Returns the predicted letter.

### aslVersion3.py

This file is an extension of `handPredict.py` with added functionality for converting ASL letters to English and other South Indian languages. It also includes an auto-correction feature to handle slight errors in recognition. The key features are:
1. It integrates the auto-correction function to ensure more accurate translations.
2. It provides an interface to translate the recognized sign into multiple languages.

### aslFinalMain.py

This file is similar to `aslVersion3.py` but without the auto-correction feature. It translates ASL gestures into English and South Indian languages, focusing solely on accurate translation without handling small errors in gesture recognition.

---

## Model Architecture

The model used for ASL recognition is a Convolutional Neural Network (CNN) built using TensorFlow/Keras. Below is a breakdown of the layers:

1. **Input Layer**: The input shape is (64, 64, 1), which represents grayscale images of size 64x64.
2. **Convolutional Layers**:
   - The first convolutional layer uses 128 filters of size (3x3) with ReLU activation.
   - The second convolutional layer uses 256 filters of size (3x3) with ReLU activation.
   - The third convolutional layer uses 256 filters of size (3x3) with ReLU activation.
   - The fourth convolutional layer uses 512 filters of size (3x3) with ReLU activation.
   - The fifth convolutional layer uses 512 filters of size (3x3) with ReLU activation.
3. **MaxPooling Layers**: After each convolutional layer, a max-pooling layer of size (2x2) is applied to down-sample the image.
4. **Dropout Layers**: Dropout layers with a rate of 0.5 are used to prevent overfitting.
5. **Flatten Layer**: The output is flattened into a one-dimensional vector.
6. **Fully Connected Layer**: A dense layer with 1024 neurons and ReLU activation is used.
7. **Output Layer**: The final output layer has 29 neurons with softmax activation to classify the 29 ASL alphabet signs.

---

## How to Run the Project

To run the ASL Translator, follow these steps:

1. **Set up the environment**:
   - Install necessary dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   
2. **Running the Translation**:
   - Run the desired Python file:
     - For letter prediction: 
       ```bash
       python handPredict.py
       ```
     - For translation with auto-correction: 
       ```bash
       python aslVersion3.py
       ```
     - For translation without auto-correction:
       ```bash
       python aslFinalMain.py
       ```

---

## Training the Model

To train the model, follow the steps below:

1. **Prepare the Data**: Collect and preprocess the ASL images. <br> You can download the dataset from: https://www.kaggle.com/grassknoted/asl-alphabet
2. **Set Up Training**:
   - Ensure the images are labeled and placed in the correct folder structure.
   - Use the `train.py` file to load and preprocess the data.
   - The model is trained using the Adam optimizer and categorical cross-entropy loss function.
   
3. **Run the Training Script**:
   ```bash
   python train.py
## Graph representation

Below is the accuracy graph for the model after training. It shows the training and validation accuracy over the epochs:

![Accuracy Graph](https://github.com/vxrnxthx/ASLtoEnglishAndOtherIndianLanguages/blob/main/Figure_1.png)

## Conclusion

The ASL Translator is an efficient solution for translating American Sign Language into readable text. The model leverages advanced deep learning techniques and can recognize hand gestures with high accuracy. With further refinement and training, it can be extended to support a wider range of gestures and languages.
