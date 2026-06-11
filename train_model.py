
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("student_placement_synthetic.csv")

# Convert text columns to numbers
le = LabelEncoder()

df['branch'] = le.fit_transform(df['branch'])
df['college_tier'] = le.fit_transform(df['college_tier'])

# Features
X = df.drop(['placement_status', 'salary_package_lpa'], axis=1)

# Target
y = df['placement_status']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

new_student = [[
    0,      # branch
    1,      # college_tier
    8.5,    # cgpa
    0,      # backlogs
    8.0,    # coding_skills
    8.0,    # dsa_score
    85,     # aptitude_score
    8.0,    # communication_skills
    7.0,    # ml_knowledge
    6.0,    # system_design
    2,      # internships
    4,      # projects_count
    3,      # certifications
    1,      # hackathons
    1,      # open_source_contributions
    1       # extracurriculars
]]

# Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

print(importance.sort_values(by='Importance', ascending=False))

# Plot
importance.sort_values(by='Importance').plot(
    x='Feature',
    y='Importance',
    kind='barh'
)

plt.show()

new_student = pd.DataFrame([[
    0,1,8.5,0,8.0,8.0,85,8.0,7.0,6.0,2,4,3,1,1,1
]], columns=X.columns)

result = model.predict(new_student)

if result[0] == 1:
    print("Likely to be Placed")
else:
    print("Not Likely to be Placed")

import joblib

joblib.dump(model, "placement_model.pkl")

print("Model saved successfully!")