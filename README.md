### **README: Chinese Herb Classifier**

---

### **Project Overview**
This project is a Convolutional Neural Network (CNN)-based image classifier to categorize images of Chinese herbs. The project includes:
- A **baseline CNN model** for initial evaluation.
- An **enhanced CNN model** for improved accuracy.
- Performance evaluation using metrics such as classification reports and confusion matrices.

---

### **Features**
1. **Image Preprocessing**: Scales and prepares images for training and validation.
2. **Baseline CNN Model**: Simple architecture for initial testing.
3. **Enhanced CNN Model**: Improved architecture with additional layers for better performance.
4. **Metrics**: Outputs accuracy, classification reports, and confusion matrices.

---

### **Setup Instructions**
1. **Install Dependencies**:
   - Install the required libraries:
     ```bash
     pip install tensorflow numpy matplotlib scikit-learn
     ```

2. **Prepare Dataset**:
   - Ensure the dataset is available as a `.zip` file (`ChineseHerbs.zip`) or extracted into the project directory.
   - The script automatically extracts the dataset if the `.zip` file is present.

3. **Run the Script**:
   - Execute the script to train and evaluate the models:
     ```bash
     python herb_classifier.py
     ```

---

### **Outputs**
1. Sample images from the dataset for visualization.
2. Performance metrics for both models:
   - Accuracy
   - Classification reports
   - Confusion matrices

---

### **Disclaimer**
This project is for **educational purposes only**. It demonstrates image classification using CNNs and is not intended for commercial or medical use.

--- 

Save this file as `README.md` in your project directory.
