# **📝 Handwritten Character Recognition using Deep Learning**  

## **📌 Overview**  
This project implements a **Handwritten Character Recognition (HCR) system** using **Deep Learning (CNNs)** to recognize handwritten letters and classify them accurately. The model is trained on a dataset of handwritten characters and predicts the correct letter from user input images.  

The system utilizes **Convolutional Neural Networks (CNNs)** to extract features from character images and predict the corresponding alphabet with high accuracy. The model is trained, validated, and tested using **Python, TensorFlow/Keras, OpenCV, and Matplotlib** in a **Jupyter Notebook environment**.  

---

## **🚀 Features**  
✔ **Real-time handwritten character recognition**  
✔ **Deep Learning-based CNN model** for high accuracy  
✔ **Preprocessing techniques** such as grayscale conversion, noise reduction, and thresholding  
✔ **Visualization of feature maps and CNN architecture**  
✔ **Graphical representation of model performance**  
✔ **Interactive interface for testing user-drawn characters**  

---

## **🛠 Tech Stack**  

| Technology       | Purpose |
|-----------------|------------------------------|
| **Python**      | Core programming language |
| **TensorFlow/Keras** | Deep Learning framework for CNN implementation |
| **OpenCV**      | Image preprocessing and visualization |
| **Matplotlib/Seaborn** | Data visualization |
| **NumPy & Pandas** | Data handling & preprocessing |
| **Jupyter Notebook** | Development environment |

---

## **📂 Dataset**  
The model is trained using a dataset of **handwritten alphabets (A-Z)**. The dataset includes:  
✔ **28x28 grayscale images** of individual characters  
✔ **Balanced class distribution** for better training  
✔ **Augmented images** to improve model generalization  

---

## **📌 Project Workflow**  

### **🔹 Step 1: Data Preprocessing**  
- Convert images to **grayscale**  
- Apply **thresholding and noise reduction**  
- Resize images to **28x28 pixels**  
- Normalize pixel values  

### **🔹 Step 2: CNN Model Architecture**  
The model consists of:  
✔ **Convolutional Layers (Conv2D)** – Extracts features from images  
✔ **Max Pooling Layers** – Reduces dimensionality while retaining features  
✔ **Fully Connected Layers (Dense)** – Classifies characters  
✔ **Softmax Activation** – Outputs probability distribution for 26 classes (A-Z)  

```python
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])
```

### **🔹 Step 3: Model Training**  
- **Loss Function**: Categorical Cross-Entropy  
- **Optimizer**: Adam  
- **Evaluation Metrics**: Accuracy, Loss  

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=20)
```

### **🔹 Step 4: Model Evaluation**  
- Accuracy: **~97.8% (Validation)**  
- Loss: **0.079 (Validation Loss)**  
- Confusion Matrix & Performance Metrics  

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()
```

---

## **🎨 Visualization & Results**  
📌 **Predicted Characters** – Displayed with confidence score  
📌 **CNN Model Architecture** – Layer-wise feature extraction  
📌 **Performance Graphs** – Accuracy vs. Loss curves  

---

## **🖍 Live Testing - User Input Handwriting**  
The project includes a **real-time testing interface** that allows users to:  
✅ Draw handwritten characters  
✅ Capture & preprocess images  
✅ Predict character with confidence score  

```python
cv2.imshow('Handwritten Character Recognition', img)
cv2.putText(img, "Prediction: " + predicted_character, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255,0,30))
```

---

## **📌 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yourusername/Handwritten-Character-Recognition.git
cd Handwritten-Character-Recognition
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Jupyter Notebook**  
```bash
jupyter notebook
```
Open `Handwritten character recognition.ipynb` and run all cells.

---

## **🚀 Future Enhancements**  
✔ Extend to **digits (0-9) and special characters**  
✔ Implement **Transformer-based models for higher accuracy**  
✔ Deploy the model as a **web app using Flask or Streamlit**  

---

🚀 **Let's make AI-powered handwriting recognition even better!**  
