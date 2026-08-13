import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('IMDb Movies India.csv', encoding='latin-1')

# 2. Drop rows where Rating is missing
df.dropna(subset=['Rating'], inplace=True)

# 3. Clean the text formatting into numbers (Fixed with raw string 'r')
df['Year'] = df['Year'].str.extract(r'(\d+)').astype(float)
df['Duration'] = df['Duration'].str.replace(r' min', '', regex=True).astype(float)
df['Votes'] = df['Votes'].str.replace(r',', '', regex=True).astype(float)

# 4. Fill in missing values with averages
df['Duration'].fillna(df['Duration'].median(), inplace=True)
df['Votes'].fillna(df['Votes'].median(), inplace=True)
df['Year'].fillna(df['Year'].median(), inplace=True)
# 5. Create reputation scores for text columns
director_mean = df.groupby('Director')['Rating'].mean().to_dict()
actor1_mean = df.groupby('Actor 1')['Rating'].mean().to_dict()
genre_mean = df.groupby('Genre')['Rating'].mean().to_dict()

df['Director_encoded'] = df['Director'].map(director_mean)
df['Actor1_encoded'] = df['Actor 1'].map(actor1_mean)
df['Genre_encoded'] = df['Genre'].map(genre_mean)

global_mean = df['Rating'].mean()
df['Director_encoded'].fillna(global_mean, inplace=True)
df['Actor1_encoded'].fillna(global_mean, inplace=True)
df['Genre_encoded'].fillna(global_mean, inplace=True)

# 6. Separate features and target
features = ['Year', 'Duration', 'Votes', 'Genre_encoded', 'Director_encoded', 'Actor1_encoded']
X = df[features]
y = df['Rating']

# 7. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train the Regressor Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluate the results
predictions = model.predict(X_test)
print("Task 2 Model R2 Score (Accuracy):", r2_score(y_test, predictions))
print("Task 2 Average Error Margin (RMSE):", np.sqrt(mean_squared_error(y_test, predictions)))