# German Credit Risk Prediction Streamlit App

📊 This project is a Streamlit web application that predicts credit risk using a **CatBoost** classifier. It allows interactive exploration of features, model evaluation, and risk prediction for custom inputs or specific test set records. The app is fully containerized with Docker and comes with unit tests and CI/CD setup.

---

## Features

- **Exploratory Data Analysis (EDA)**
  - Visualizations for numeric and categorical features
  - Stacked histograms and bar charts showing distribution by risk
  - Correlation inspection and feature importance visualization

- **Model Training & Evaluation**
  - CatBoost classifier trained on the German Credit dataset
  - Train/Validation split metrics: accuracy, confusion matrix, classification report
  - Feature importance display (top 15 features)

- **Prediction**
  - **Custom Input Mode:** Enter your own values to see predicted risk
  - **Test Set ID Mode:** Select a record from the test dataset by its ID for prediction
  - Results include predicted risk and corresponding feature values

- **Dockerized Deployment**
  - Run the app locally or pull the image from DockerHub
  - CI/CD workflow automatically runs tests, builds Docker image, and pushes to DockerHub

---