import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import streamlit as st

# Load Dataset
data_path = "synthetic_resume_dataset.csv"
df = pd.read_csv(data_path)

# Preprocessing
def preprocess_data(df):
    # Combine text fields for vectorization
    df['CombinedText'] = df['Skills'] + ' ' + df['Education'] + ' ' + df['Certifications'] + ' ' + df['Projects']
    vectorizer = TfidfVectorizer(max_features=500)
    text_features = vectorizer.fit_transform(df['CombinedText']).toarray()

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(df[['Experience(Years)']])

    # Combine all features
    features = np.hstack((text_features, numerical_features))
    return features, vectorizer, scaler

features, vectorizer, scaler = preprocess_data(df)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(features)
df['Cluster'] = cluster_labels

# Naming Clusters
cluster_names = {0: 'Payroll', 1: 'Absences', 2: 'Compensation', 3: 'Core HR'}
df['Module'] = df['Cluster'].map(cluster_names)

# Save Model Details
def predict_module(user_input):
    # Vectorize user input
    text_features = vectorizer.transform([user_input['CombinedText']]).toarray()
    numerical_features = scaler.transform([[user_input['Experience(Years)']]])
    input_features = np.hstack((text_features, numerical_features))

    # Predict Cluster
    cluster = kmeans.predict(input_features)[0]
    return cluster_names[cluster]

# Streamlit App
st.title("Resume Clustering and Module Prediction")

st.header("Enter Candidate Details")
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=1, step=1)
skills = st.text_area("Skills")
education = st.text_input("Education")
certifications = st.text_input("Certifications")
projects = st.text_area("Projects")

if st.button("Predict Module"):
    if not (skills and education and certifications and projects):
        st.error("Please fill out all fields!")
    else:
        # Combine user inputs
        user_input = {
            'Experience(Years)': experience,
            'CombinedText': skills + ' ' + education + ' ' + certifications + ' ' + projects
        }
        module = predict_module(user_input)
        st.success(f"The predicted module is: {module}")
