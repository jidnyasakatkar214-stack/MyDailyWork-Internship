import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the data
df = pd.read_csv('tested.csv')

# 2. Clean the data
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# 3. Convert text to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].astype('category').cat.codes

# Drop any remaining missing values
df = df.dropna()

# 4. Separate features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# 5. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Print the result
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))