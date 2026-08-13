# Iris Flower Classification using K-Nearest Neighbors

## 📌 MyDailyWork Internship – Task 3

This project was completed as part of my **MyDailyWork Internship**.

The objective of this task is to build a Machine Learning classification model that identifies the species of an Iris flower based on its sepal and petal measurements.

---

## 🎯 Objective

The main objectives of this task are:

* Load the Iris dataset.
* Select relevant flower features.
* Split the dataset into training and testing sets.
* Standardize the feature values.
* Train a K-Nearest Neighbors (KNN) classification model.
* Evaluate the model using accuracy and a classification report.
* Predict the species of a new sample.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

### Machine Learning Algorithm

**K-Nearest Neighbors (KNN)**

---

## 📂 Dataset

The project uses a dataset named:

```text
IRIS.csv
```

The dataset contains four numerical features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The target variable is:

```text
species
```

These four features are explicitly selected in the Python program.

---

## 🔄 Project Workflow

### 1. Load the Dataset

The program loads `IRIS.csv` using Pandas and handles both UTF-8 and Latin-1 encoding.

### 2. Select Features and Target

The following features are used:

```text
sepal_length
sepal_width
petal_length
petal_width
```

The target variable is `species`.

### 3. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratification is used to maintain the class distribution between training and testing data.

### 4. Feature Scaling

The features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Train the KNN Model

A **K-Nearest Neighbors Classifier** is created with:

```text
n_neighbors = 3
```

The model is trained using the scaled training data.

### 6. Make Predictions

The trained model predicts the Iris species for the test dataset.

### 7. Evaluate the Model

The model is evaluated using:

* **Accuracy Score**
* **Classification Report**

The program prints the model accuracy as a percentage.

### 8. Sample Prediction

The project also tests the trained model using the sample:

```text
[5.1, 3.5, 1.4, 0.2]
```

The sample is scaled using the same scaler before the model predicts its species.

---

## 📊 Model Evaluation

The program displays:

```text
--- Accuracy Score ---
Model Accuracy: XX.XX%

--- Classification Report ---
...
```

The exact accuracy depends on the `IRIS.csv` dataset used during execution.

---

## 🌸 Sample Prediction

For the input:

```text
[5.1, 3.5, 1.4, 0.2]
```

the model predicts the corresponding Iris species.

---

## 📁 Project Structure

```text
Iris-Flower-Classification/
│
├── main.py
├── IRIS.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Add the Dataset

Place `IRIS.csv` in the same directory as `main.py`.

### 3. Run the Program

```bash
python main.py
```

### 4. View the Results

The program will display:

* First five rows of the dataset
* Model accuracy
* Classification report
* Sample prediction

---

## 📚 Key Learning Outcomes

Through this task, I gained practical experience in:

* Loading datasets using Pandas
* Feature selection
* Train-test splitting
* Feature standardization
* KNN classification
* Model prediction
* Accuracy evaluation
* Classification reports
* Predicting new data samples

---

## 👩‍💻 Internship Information

**Internship:** MyDailyWork
**Task:** Task 3 – Iris Flower Classification
**Domain:** Machine Learning / Data Science
**Language:** Python
**Algorithm:** K-Nearest Neighbors (KNN)

---

## ✅ Conclusion

This project demonstrates a complete Machine Learning classification workflow using the Iris dataset. The flower measurements are standardized and provided to a **K-Nearest Neighbors classifier** to identify the species. The model is evaluated using accuracy and a classification report and is also used to classify a new sample.
