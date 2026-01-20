# 🐶🐱 Cat-And-Dog-Classifier

A **Convolutional Neural Network (CNN)** based deep learning project to classify images of **Cats and Dogs**. The model is trained using TensorFlow/Keras and deployed with a simple **Gradio web interface** that allows users to upload images and get real-time predictions.

---

## 📌 Project Overview

Image classification is one of the most popular applications of **Computer Vision**. In this project, a CNN model is built to:

* Learn visual features of cats and dogs
* Classify uploaded images as **Cat** or **Dog**
* Provide predictions through a user-friendly web interface

This project is suitable for **beginners to intermediate learners** in Deep Learning and is also **interview & portfolio ready**.

---

## 🧠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **CNN (Convolutional Neural Network)**
* **NumPy**
* **Pillow (PIL)**
* **Gradio** (for deployment)

---

## 📂 Project Structure


Cat-And-Dog-Classifier/
│── app.py                 # Gradio web app
│── model.h5               # Trained CNN model
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation


---

## 🏗️ Model Architecture

The CNN model includes:

* Convolutional layers for feature extraction
* MaxPooling layers for dimensionality reduction
* Fully connected (Dense) layers for classification
* Sigmoid activation for binary classification

**Loss Function:** Binary Crossentropy
**Optimizer:** Adam
**Evaluation Metrics:** Accuracy, Precision, Recall

---

## 📊 Dataset

* Dataset contains images of **Cats** and **Dogs**
* Images are resized to a fixed input size before training
* Data is normalized by rescaling pixel values (0–1)

> Dataset source: Common Kaggle Cats vs Dogs dataset

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Cat-And-Dog-Classifier.git
cd Cat-And-Dog-Classifier
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Gradio App

```bash
python app.py
```

### 4️⃣ Open in Browser

After running the app, open the generated **local or public URL** and upload an image to get predictions.

---

## 🌐 Deployment

The project can be deployed publicly using:

* **Hugging Face Spaces (Gradio)** ✅ *(Recommended)*
* Local deployment with `interface.launch(share=True)`

---

## 🖼️ Sample Output

* Upload a cat image → **🐱 Cat (Confidence %)**
* Upload a dog image → **🐶 Dog (Confidence %)**

---

## 🎯 Use Cases

* Learning CNN fundamentals
* Image classification projects
* ML/DL portfolio showcase
* Interview demonstrations

---

## 🔮 Future Improvements

* Multi-class image classification
* Improve accuracy with data augmentation
* Add confidence bar visualization
* Deploy using Streamlit
* Convert model to TensorFlow Lite (TFLite)

---

## 👨‍💻 Author

**Dnyaneshwar Kale**
B.E. Computer Science | Data Analyst | ML Enthusiast

---

## ⭐ Acknowledgements

* TensorFlow Documentation
* Gradio Team
* Kaggle Dataset Contributors

---

⭐ If you like this project, don’t forget to **star the repository**!

