import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 1: Load and Parse the JSON Data
with open('corrected_data.json', 'r') as file:
    json_data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame(json_data)

# Step 2: Extract Features and Labels
X = df['transcript']
y_tool = df['tool']
y_heading = df['heading']
y_target = df['target']

# Step 3: Encode Labels
encoder_tool = LabelEncoder()
encoder_heading = LabelEncoder()
encoder_target = LabelEncoder()

y_tool_encoded = encoder_tool.fit_transform(y_tool)
y_heading_encoded = encoder_heading.fit_transform(y_heading)
y_target_encoded = encoder_target.fit_transform(y_target)

# Combine the labels into a single DataFrame
y_combined = pd.DataFrame({
    'tool': y_tool_encoded,
    'heading': y_heading_encoded,
    'target': y_target_encoded
})

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)

# Step 5: Create a Pipeline for Preprocessing and Model Training
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

# Step 6: Train the Model
pipeline.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = pipeline.predict(X_test)

# Evaluate each output separately
print("Classification Report for 'tool':")
print(classification_report(y_test['tool'], y_pred[:, 0], target_names=encoder_tool.classes_, zero_division=1))

print("Classification Report for 'heading':")
print(classification_report(y_test['heading'], y_pred[:, 1], zero_division=1))

print("Classification Report for 'target':")
print(classification_report(y_test['target'], y_pred[:, 2], labels=encoder_target.transform(encoder_target.classes_), target_names=encoder_target.classes_, zero_division=1))

# Step 8: Save the Model and Encoders
joblib.dump(pipeline, 'nlp_model.pkl')
joblib.dump(encoder_tool, 'encoder_tool.pkl')
joblib.dump(encoder_heading, 'encoder_heading.pkl')
joblib.dump(encoder_target, 'encoder_target.pkl')

print("Model and encoders have been successfully saved.")

# Step 9: Function for Prediction
def predict_transcript(transcript):
    # Load the trained model and encoders
    model = joblib.load('nlp_model.pkl')
    encoder_tool = joblib.load('encoder_tool.pkl')
    encoder_heading = joblib.load('encoder_heading.pkl')
    encoder_target = joblib.load('encoder_target.pkl')
    
    # Make a prediction
    prediction = model.predict([transcript])
    tool_pred = encoder_tool.inverse_transform([prediction[0][0]])[0]
    heading_pred = encoder_heading.inverse_transform([prediction[0][1]])[0]
    target_pred = encoder_target.inverse_transform([prediction[0][2]])[0]
    
    return {
        'tool': tool_pred,
        'heading': heading_pred,
        'target': target_pred
}

# Example usage
example_transcript = "Heading is one five zero, target is green commercial aircraft, tool to deploy is electromagnetic pulse."
result = predict_transcript(example_transcript)
print(result)
def main():
    while True:
        transcript = input("Enter a transcript (or 'exit' to quit): ")
        if transcript.lower() == 'exit':
            break
        result = predict_transcript(transcript)
        print("Predicted Result:", result)

if __name__ == "__main__":
    main()