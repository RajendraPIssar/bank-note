import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Function to normalize input data
def normalize_input(value, mean, min_val, max_val):
    return (value - mean) / (max_val - min_val)

# Load the data and perform initial preprocessing
@st.cache
def load_data():
    data = pd.read_csv("banknotes.csv")
    data.dropna(inplace=True)
    return data

# Train the KMeans model
def train_model(data):
    x = data['V1']
    y = data['V2']

    # Normalize the data
    mean_x = x.mean()
    mean_y = y.mean()
    max_x = x.max()
    max_y = y.max()
    min_x = x.min()
    min_y = y.min()

    # Normalizing data
    x_norm = (x - mean_x) / (max_x - min_x)
    y_norm = (y - mean_y) / (max_y - min_y)

    # Train KMeans with 2 clusters
    model = KMeans(n_clusters=2, max_iter=243).fit(np.column_stack((x_norm, y_norm)))
    
    return model, mean_x, mean_y, min_x, min_y, max_x, max_y

# Streamlit App UI
def main():
    st.title("Banknote Genuinity Prediction")

    # Explanation of V1 and V2
    st.subheader("What are V1 and V2?")
    st.write("**V1 (Variance of Wavelet Transformed Image)**: This feature represents the variance (spread) of pixel intensities in the wavelet-transformed image of the banknote. Positive values indicate more variance.")
    st.write("**V2 (Skewness of Wavelet Transformed Image)**: This feature represents the skewness (asymmetry) of pixel intensities in the wavelet-transformed image. Values can be either positive or negative.")

    st.write("Enter the values for V1 and V2 to predict if a banknote is fake or not.")

    # Input fields for user to enter V1 and V2 values
    v1 = st.number_input("V1 (Variance) - Enter positive values only", min_value=0.0, format="%.2f")
    v2 = st.number_input("V2 (Skewness) - Enter any value", format="%.2f")

    # Load data and train model
    data = load_data()
    model, mean_x, mean_y, min_x, min_y, max_x, max_y = train_model(data)

    # When user clicks the "Predict" button
    if st.button("Predict"):
        # Normalize user inputs
        v1_normalized = normalize_input(v1, mean_x, min_x, max_x)
        v2_normalized = normalize_input(v2, mean_y, min_y, max_y)

        # Predict the cluster (0 or 1)
        prediction = model.predict([[v1_normalized, v2_normalized]])

        # Output prediction result
        if prediction[0] == 0:
            st.success("The banknote is Genuine!")
        else:
            st.error("The banknote is Fake!")

if __name__ == "__main__":
    main()
