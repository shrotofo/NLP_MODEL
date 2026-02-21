# NLP Multi-Label Classification Model

## Features
**Multi-Label Classification** : Classifies transcripts into multiple categories: tool, heading, and target.

## Technologies Used
**ML Model** : Scikit-learn with a multi-output logistic regression classifier

## How It Works
**Model Architecture** 
The model uses a logistic regression classifier wrapped in a multi-output classifier to handle multiple target variables simultaneously. This allows it to classify each transcript into three separate categories.

## Steps Involved
*Data Preparation*

**Loading Data**: The data is loaded from a JSON file and converted into a pandas DataFrame.

**Extracting Features and Labels**: The transcripts are extracted as features, and the corresponding tool, heading, and target labels are extracted.

**Encoding Labels**: The categorical labels for tool, heading, and target are encoded into numerical values using LabelEncoder.

*Loading and Parsing the JSON Data*

The dataset is parsed from a JSON file and loaded into a pandas DataFrame, making it easy to manipulate and analyze.

*Splitting the Data*

**Train-Test Split**: The dataset is split into training and testing sets to evaluate the model's performance on unseen data. This ensures that the model is not overfitting and can generalize well.

*Creating a Pipeline*

**Vectorization**: Text data (transcripts) are converted into numerical vectors using CountVectorizer.

**Model Training**: A logistic regression classifier is used within a MultiOutputClassifier to handle the multi-label nature of the task. The pipeline handles both vectorization and classification.

*Training the Model*

**Pipeline Training**: The pipeline is fitted on the training data, where the model learns to map transcripts to their corresponding labels.

**Handling Multiple Outputs**: The multi-output classifier ensures that each label (tool, heading, target) is predicted separately but simultaneously.

*Evaluation*

**Model Prediction**: The model predicts labels for the test set, and the performance is evaluated.

**Classification Report**: A classification report is generated for each label, showing metrics like precision, recall, and F1-score, providing insights into the model's performance.

*Saving the Model and Encoders*

**Persistence**: The trained model and the label encoders are saved to disk using joblib, enabling easy reuse and deployment without retraining.

*Prediction*

**Loading the Model** : The saved model and encoders are loaded.

**Making Predictions** : The predict_transcript function takes a new transcript, processes it, and predicts the tool, heading, and target labels. The function returns the predicted labels as a dictionary.
