# Movie Rating Prediction using Machine Learning

## 📌 MyDailyWork Internship – Task 2

This project was completed as part of my **MyDailyWork Internship**.

The objective of this task is to build a Machine Learning model that predicts **IMDb movie ratings** using information such as year, duration, votes, genre, director, and actor.

---

## 🎯 Objective

The main objective of this project is to:

* Load and preprocess the IMDb Movies India dataset.
* Handle missing values.
* Convert text-based numerical data into usable numerical features.
* Encode categorical information such as genre, director, and actor.
* Train a Random Forest Regression model.
* Predict movie ratings.
* Evaluate the model using **R² Score** and **RMSE**.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Random Forest Regression**

---

## 📂 Dataset

The project uses:

```text
IMDb Movies India.csv
```

The dataset contains movie-related information used to predict the movie's rating.

The program removes records where the `Rating` value is missing before training the model.

---

## 🔄 Project Workflow

### 1. Load the Dataset

The dataset is loaded using Pandas with `latin-1` encoding.

```python
df = pd.read_csv('IMDb Movies India.csv', encoding='latin-1')
```

### 2. Data Cleaning

Rows with missing ratings are removed.

The following columns are converted into numerical values:

* `Year`
* `Duration`
* `Votes`

For example, `"120 min"` is converted into a numerical duration value and commas are removed from vote counts.

### 3. Handling Missing Values

Missing values in:

* Duration
* Votes
* Year

are replaced using their respective median values.

### 4. Feature Encoding

The project creates rating-based encoded features for:

* Director
* Actor 1
* Genre

The mean rating associated with each director, actor, and genre is calculated and mapped back to the dataset.

Missing encoded values are replaced with the global mean rating.

### 5. Feature Selection

The following features are used by the model:

```text
Year
Duration
Votes
Genre_encoded
Director_encoded
Actor1_encoded
```

The target variable is:

```text
Rating
```

### 6. Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7. Model Training

A **Random Forest Regressor** with 100 estimators is used to predict movie ratings.

```python
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
```

### 8. Model Evaluation

The model generates predictions for the test dataset.

Two evaluation metrics are calculated:

* **R² Score**
* **RMSE (Root Mean Squared Error)**

```python
predictions = model.predict(X_test)

r2_score(y_test, predictions)

np.sqrt(mean_squared_error(y_test, predictions))
```

---

## 📊 Output

The program displays:

```text
Task 2 Model R2 Score (Accuracy): <value>
Task 2 Average Error Margin (RMSE): <value>
```

The exact values depend on the dataset used when the program is executed.

---

## 📁 Project Structure

```text
Movie-Rating-Prediction/
│
├── movie_main.py
├── IMDb Movies India.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy scikit-learn
```

### 2. Add the Dataset

Place `IMDb Movies India.csv` in the same folder as `movie_main.py`.

### 3. Run the Program

```bash
python movie_main.py
```

### 4. View the Results

The terminal will display the model's **R² Score** and **RMSE**.

---

## 📚 Key Learning Outcomes

Through this task, I gained practical experience in:

* Data preprocessing
* Handling missing values
* Feature engineering
* Numerical data extraction
* Categorical feature encoding
* Train-test splitting
* Random Forest Regression
* Model prediction
* R² Score evaluation
* RMSE evaluation

---

## 👩‍💻 Internship Information

**Internship:** MyDailyWork
**Task:** Task 2 – Movie Rating Prediction
**Domain:** Machine Learning / Data Science
**Language:** Python
**Model:** Random Forest Regressor

---

## ✅ Conclusion

This project demonstrates a complete Machine Learning regression workflow for predicting movie ratings. The IMDb Movies India dataset is cleaned and transformed into numerical features before being used to train a **Random Forest Regressor**. The model's performance is evaluated using **R² Score and RMSE**.
