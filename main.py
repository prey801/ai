import pandas as pd
import numpy as np
import pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import custom modules
from feature_extraction import extract_url_features, extract_email_features
from model import train_traditional_models, train_neural_network, create_ensemble_model

#Data cleaning & preprocessing
def preprocess_url_data(df):
    # Handle missing values
    df = df.dropna(subset=['url', 'is_phishing'])

    # Convert label to binary
    df['is_phishing'] = df['is_phishing'].astype(int)

    return df

def main():
    print("Starting phishing detection model training...")

    # Load datasets
    print("Loading datasets...")
    phishtank_data = pd.read_csv('phishtank_dataset.csv')
    uci_data = pd.read_csv('uci_phishing_dataset.csv')
    enron_data = pd.read_csv('enron_email_dataset.csv')

    # Process URL data
    print("Processing URL data...")
    url_data = pd.concat([preprocess_url_data(phishtank_data),
                         preprocess_url_data(uci_data)])

    # Extract URL features
    print("Extracting URL features...")
    url_features = []
    url_labels = []

    for index, row in url_data.iterrows():
        if index % 1000 == 0:
            print(f"Processing URL {index}/{len(url_data)}")
        try:
            features = extract_url_features(row['url'])
            url_features.append(list(features.values()))
            url_labels.append(row['is_phishing'])
        except Exception as e:
            print(f"Error processing URL {row['url']}: {e}")
            continue

    # Convert to DataFrame
    X_url = pd.DataFrame(url_features, columns=list(extract_url_features(url_data.iloc[0]['url']).keys()))
    y_url = np.array(url_labels)

    # Split URL data
    X_url_train, X_url_test, y_url_train, y_url_test = train_test_split(
        X_url, y_url, test_size=0.2, random_state=42, stratify=y_url
    )

    # Define numerical and categorical features
    numerical_features = X_url_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['tld']  # Assuming 'tld' is the only categorical feature

    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Apply preprocessing to training and test data
    X_url_train = preprocessor.fit_transform(X_url_train)
    X_url_test = preprocessor.transform(X_url_test)

    # Train URL models
    print("Training URL models...")
    url_models, url_results = train_traditional_models(X_url_train, y_url_train, X_url_test, y_url_test)

    # Print URL model results
    print("\nURL Model Results:")
    for model_name, metrics in url_results.items():
        print(f"{model_name}: {metrics}")

    # Select best URL model (Random Forest typically performs well for this task)
    best_url_model = url_models['random_forest']

    # Process email data
    print("\nProcessing email data...")
    # For simplicity, let's sample a portion of the Enron dataset
    sample_size = min(5000, len(enron_data))
    email_sample = enron_data.sample(n=sample_size, random_state=42)

    # Create synthetic phishing emails (in a real scenario, you'd have actual phishing emails)
    # Here we're just simulating by marking emails with certain keywords as phishing
    email_sample['is_phishing'] = email_sample['content'].str.contains(
        'password|urgent|verify|account|suspend|confirm', case=False).astype(int)

    # Extract email features
    print("Extracting email features...")
    email_features = []
    email_labels = []

    for index, row in email_sample.iterrows():
        if index % 500 == 0:
            print(f"Processing email {index}/{len(email_sample)}")
        try:
            features = extract_email_features(row['content'])
            # Convert any non-numeric features to numeric
            feature_values = []
            for key, value in features.items():
                if isinstance(value, str):
                    # Skip or encode string features
                    continue
                feature_values.append(value)

            email_features.append(feature_values)
            email_labels.append(row['is_phishing'])
        except Exception as e:
            print(f"Error processing email at index {index}: {e}")
            continue

    # Convert to numpy arrays
    X_email = np.array(email_features)
    y_email = np.array(email_labels)

    # Split email data
    # Split email data, but only if we have samples
    if len(X_email) > 0:
      X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(X_email, y_email, test_size=0.2, random_state=42, stratify=y_email)
    else:
      # Handle empty dataset case
      X_email_train, X_email_test, y_email_train, y_email_test = [], [], [], []

    # Normalize features
    email_scaler = StandardScaler()
    X_email_train = email_scaler.fit_transform(X_email_train)
    X_email_test = email_scaler.transform(X_email_test)

    # Train email models
    print("Training email models...")
    email_models, email_results = train_traditional_models(X_email_train, y_email_train, X_email_test, y_email_test)

    # Print email model results
    print("\nEmail Model Results:")
    for model_name, metrics in email_results.items():
        print(f"{model_name}: {metrics}")

    # Select best email model
    best_email_model = email_models['random_forest']

    # Define scaler for URL pipeline
    scaler = StandardScaler()

    # Create pipelines to include preprocessing steps
    url_pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', best_url_model)
    ])
    url_pipeline.fit(X_url_train, y_url_train)

    email_pipeline = Pipeline([
        ('scaler', email_scaler),
        ('classifier', best_email_model)
    ])
    email_pipeline.fit(X_email_train, y_email_train)

    # Save the models
    print("\nSaving models...")
    pickle.dump(url_pipeline, open('url_model.pkl', 'wb'))
    pickle.dump(email_pipeline, open('email_model.pkl', 'wb'))

    print("Models saved successfully!")
    print("URL model saved as 'url_model.pkl'")
    print("Email model saved as 'email_model.pkl'")

if __name__ == "__main__":
    main()