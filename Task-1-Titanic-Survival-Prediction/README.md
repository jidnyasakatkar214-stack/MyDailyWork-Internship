# Titanic Survival Prediction using Machine Learning

## 📌 Internship Task

This project was completed as part of my **MyDailyWork Internship**. The objective of this task is to build a Machine Learning classification model that predicts whether a passenger survived the Titanic disaster.

## 🎯 Objective

The main objective is to:

* Load and preprocess the Titanic dataset.
* Handle missing values.
* Convert categorical data into numerical form.
* Separate features and target variables.
* Train a Machine Learning classification model.
* Predict passenger survival.
* Evaluate the model using accuracy.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Random Forest Classifier**

## 📂 Dataset

The project uses a CSV file named:

```text
tested.csv
```

The dataset contains passenger-related information, including features such as:

* Passenger ID
* Passenger Class
* Name
* Sex
* Age
* Fare
* Embarked
* Survival status

## 🔄 Project Workflow

### 1. Data Loading

The Titanic dataset is loaded using Pandas:

```python
df = pd.read_csv('tested.csv')
```

### 2. Data Cleaning

Unnecessary columns such as `PassengerId`, `Name`, `Ticket`, and `Cabin` are removed.

Missing values in `Age` and `Fare` are replaced with their respective median values.

```python
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
```

### 3. Categorical Data Encoding

The `Sex` column is converted into numerical values:

* Male → `0`
* Female → `1`

The `Embarked` column is also converted into numerical category codes.

```python
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].astype('category').cat.codes
```

### 4. Feature and Target Separation

The `Survived` column is used as the target variable, while the remaining columns are used as input features.

```python
X = df.drop(columns=['Survived'])
y = df['Survived']
```

### 5. Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 6. Model Training

A **Random Forest Classifier** is used for prediction.

```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 7. Model Evaluation

The trained model predicts survival for the test dataset, and its performance is measured using accuracy.

```python
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))
```

## 📊 Output

The program displays the model's accuracy in the following format:

```text
Model Accuracy: <accuracy value>
```

The exact accuracy depends on the contents of the `tested.csv` dataset used when running the program.

## 📁 Project Structure

```text
Titanic-Survival-Prediction/
│
├── main.py
├── tested.csv
└── README.md
```

## 🚀 How to Run

### 1. Install Python

Make sure Python is installed on your system.

### 2. Install Required Libraries

```bash
pip install pandas scikit-learn
```

### 3. Add the Dataset

Place `tested.csv` in the same directory as `main.py`.

### 4. Run the Program

```bash
python main.py
```

## 📚 Key Learning Outcomes

Through this task, I gained practical experience in:

* Data loading using Pandas
* Data preprocessing
* Handling missing values
* Categorical data encoding
* Feature and target selection
* Train-test splitting
* Random Forest classification
* Model prediction
* Model accuracy evaluation

## 👩‍💻 Internship

**Internship:** MyDailyWork
**Task:** Titanic Survival Prediction
**Domain:** Machine Learning / Data Science
**Language:** Python

## ✅ Conclusion

This task demonstrates a basic Machine Learning workflow, starting from dataset preprocessing and continuing through model training and evaluation. A Random Forest Classifier was implemented to predict Titanic passenger survival based on the available passenger features.
