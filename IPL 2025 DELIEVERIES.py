import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv(r"C:\Users\hassa\OneDrive\Documents\Visual Studio 2022\ML PROJECTS\ipl_2025_deliveries.csv")

print(df.head())
print(df.info())

df = df.dropna(subset=['wicket_type'])

df['batting_team'] = pd.Categorical(df['batting_team']).codes
df['bowling_team'] = pd.Categorical(df['bowling_team']).codes
df['wicket_type'] = pd.Categorical(df['wicket_type']).codes

df = df[['batting_team', 'bowling_team', 'over', 'runs_of_bat', 'extras', 'wicket_type']]

X = df[['batting_team', 'bowling_team', 'over', 'runs_of_bat', 'extras']]
y = df['wicket_type']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.show()

example_delivery = [[2, 3, 5, 10, 0]]
predicted_wicket_type = model.predict(example_delivery)
print(f"Predicted Wicket Type: {predicted_wicket_type[0]}")
