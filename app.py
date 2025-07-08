import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Streamlit page setup
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🛍️ Customer Segmentation using K-Means Clustering")

# Upload CSV
st.subheader("📂 Upload the Mall Customers Dataset")
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)

    st.subheader("👀 Data Preview")
    st.write(customer_data.head())

    st.subheader("📊 Dataset Info")
    st.text(f"Shape: {customer_data.shape}")
    st.text("Missing Values:\n" + str(customer_data.isnull().sum()))

    st.subheader("📌 K-Means Clustering (on Annual Income & Spending Score)")
    X = customer_data.iloc[:, [3, 4]].values

    # Elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    st.subheader("📈 Elbow Method to Determine Optimal Clusters")
    fig1, ax1 = plt.subplots()
    sns.set()
    ax1.plot(range(1, 11), wcss, marker='o')
    ax1.set_title('The Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS')
    st.pyplot(fig1)

    # Fit KMeans with optimal k (5 here)
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    Y = kmeans.fit_predict(X)

    # Visualize clusters
    st.subheader("🎯 Customer Clusters")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors = ['green', 'red', 'yellow', 'violet', 'blue']
    labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

    for i in range(5):
        ax2.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=labels[i])

    # Centroids
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='cyan', label='Centroids', marker='X')
    ax2.set_title('Customer Groups by K-Means Clustering')
    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Spending Score (1-100)')
    ax2.legend()
    st.pyplot(fig2)

else:
    st.info("Please upload the 'Mall_Customers.csv' dataset to begin.")
