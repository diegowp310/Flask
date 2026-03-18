import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Select features and target
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']
target = 'survived'

df = df[features + [target]].dropna()

# Encode sex: male=0, female=1
df['sex'] = df['sex'].map({'male': 0, 'female': 1})

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model.pkl saved successfully.")
