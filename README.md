

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

This project is an **American Sign Language (ASL) Translator** that uses a **pre-trained Convolutional Neural Network (CNN) model** to recognize hand gestures and translate them into **English and South Indian languages**. It also includes features such as **text-to-speech (TTS) conversion**, **auto-correction of recognized words**, and a **GUI-based interface**.  

The system utilizes **MediaPipe** for real-time hand tracking and **Google Translate API** for language conversion.  

---

## File Descriptions  

### handPredict.py  

This file handles the core **ASL letter recognition**. It:  
1. Loads the pre-trained model (`model_trained.h5`).  
2. Uses **MediaPipe** to track hand landmarks.  
3. Extracts the hand region and processes it using **edge detection**.  
4. Predicts the corresponding ASL letter using the CNN model.  

---

### aslVersion3.py  

This file extends `handPredict.py` by adding:  
1. **Multi-language translation** using the Google Translate API.  
2. **Auto-correction of words** using the `Auto_Correct_SpellChecker` module.  
3. **Text-to-Speech (TTS)** conversion using `gTTS` and `pygame`.  
4. A **dark-themed GUI** for user-friendly interaction.  

The GUI allows users to:  
- See recognized ASL letters.  
- Form words and sentences.  
- Translate sentences into **multiple Indian languages**.  
- Hear the translated text using the **Text-to-Speech feature**.  

---

### aslFinalMain.py  

This file is similar to `aslVersion3.py` but **does not include auto-correction**.  
- It recognizes ASL letters and translates them into multiple languages.  
- Provides real-time gesture recognition and translation.  
- Includes a **dark-themed user interface**.  

---

## Model Architecture  

The ASL recognition model is a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras**.  

### **Model Layers**  
1. **Input Layer**: 64×64 grayscale images of ASL signs.  
2. **Convolutional Layers**:  
   - 128 filters (3×3) – ReLU  
   - 256 filters (3×3) – ReLU  
   - 256 filters (3×3) – ReLU  
   - 512 filters (3×3) – ReLU  
   - 512 filters (3×3) – ReLU  
3. **MaxPooling Layers**: (2×2) after each convolutional layer.  
4. **Dropout Layers**: Dropout rate of **0.5** to prevent overfitting.  
5. **Flatten Layer**: Converts extracted features into a **1D vector**.  
6. **Fully Connected Layer**: 1024 neurons, ReLU activation.  
7. **Output Layer**: 29 neurons (one for each ASL alphabet sign) with **softmax activation**.  

---

## How to Run the Project  

### **1. Set Up the Environment**  
Install the required dependencies using:  
```bash
pip install -r requirements.txt
```

### **2. Run the Application**  

- **For ASL letter prediction only:**  
  ```bash
  python handPredict.py
  ```
- **For ASL translation with auto-correction:**  
  ```bash
  python aslVersion3.py
  ```
- **For ASL translation without auto-correction:**  
  ```bash
  python aslFinalMain.py
  ```

---

## Training the Model  

### **1. Prepare the Dataset**  
Download the ASL alphabet dataset:  
[ASL Alphabet Dataset - Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet)  

Ensure the dataset is structured correctly before training.  

### **2. Train the Model**  
Run the training script:  
```bash
python train.py
```
This will train a CNN model to recognize ASL signs.  

---

## Graphical Representation  

Below is the **training accuracy graph** for the model, showing how well the model improves over epochs:  

![Accuracy Graph](https://github.com/vxrnxthx/ASLtoEnglishAndOtherIndianLanguages/blob/main/Figure_1.png)  

---

## Conclusion  

This **ASL Translator** is an efficient solution for converting sign language into readable text and speech. It integrates **deep learning, real-time image processing, language translation, and text-to-speech technologies** to create a highly interactive and useful application.  

With further refinements, the model can be **expanded to recognize full words and sentences**, making it an invaluable tool for **deaf and mute communities** worldwide.  
