import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle 

print("Loading dataset...")
try:
    df = pd.read_csv('engagement_dataset.csv')
except FileNotFoundError:
    print("ERROR: Run Step 1 first!")
    exit()

X = df.drop('class', axis=1) # Features
y = df['class'] # Labels

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# Build Pipeline
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())

print("Training model... (This might take a moment)")
model = pipeline.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save Model
with open('engagement_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("SUCCESS! 'engagement_model.pkl' created.")