# Fashion Recommendation System

## 🚀 Overview
The **Fashion Recommendation System** is an AI-powered tool that helps users find visually similar fashion items using image-based search. By leveraging deep learning and computer vision, this system enhances the online shopping experience by providing personalized fashion recommendations.

## 📌 Features
- **Image-based search**: Upload an image to find similar fashion items.
- **Deep Learning-powered recommendations** using ResNet50.
- **K-Nearest Neighbors (KNN) for similarity search.**
- **Web-based UI** using Streamlit.
- **Deployed App**: [Try it here](https://image-based-recom-model-afzqvwnxaabctvxcagdpzh.streamlit.app/).

## 🛠️ Tech Stack
- **Deep Learning Model:** ResNet50 (Feature Extraction)
- **Machine Learning Algorithm:** K-Nearest Neighbors (KNN) for similarity search
- **Frameworks & Tools:**
  - TensorFlow & Keras
  - Scikit-Learn
  - Streamlit (Web UI)
  - Python

## 🔧 Installation
### **1️⃣ Clone the Repository**
```sh
 git clone https://github.com/hassanyusuf6826/Image-based-recom-model.git
 cd Fashion-Recommendation-System
```

### **2️⃣ Install Dependencies**
```sh
 pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```sh
 streamlit run app.py
```

## 🎯 How It Works
1. **Upload an Image**: The system accepts fashion images as input.
2. **Feature Extraction**: The image is processed using ResNet50 to extract feature vectors.
3. **Similarity Search**: The extracted features are compared with a precomputed database using KNN.
4. **Display Recommendations**: The system returns the most visually similar fashion items.

## 🌍 Deployment
The project is deployed using **Streamlit Cloud**. Access the live version here:
[🔗 Click to Try the App](https://image-based-recom-model-afzqvwnxaabctvxcagdpzh.streamlit.app/)

## 🏆 Future Enhancements
- Improve model accuracy with **fine-tuned CNNs**.
- Add **multi-modal recommendations** (text & image search).
- Integrate with **real-time e-commerce databases**.
- Deploy as a **mobile-friendly application**.

## 🤝 Contributing
Contributions are welcome! Feel free to **fork** this repository and submit a **pull request**.

## 📜 License
This project is licensed under the **MIT License**.

---
**Author:** Yusuf Hassan | **Machine Learning Intern** | **Zummit Africa**
