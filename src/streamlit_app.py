import streamlit as st
import pandas as pd
import pickle
import joblib
from sklearn.metrics import accuracy_score

from components.data_ingestion import DataIngestion
from components.data_pipeline import DataTransformationConfig, DataPipeline
from components.model_training import ModelTrainer, ModelTrainerConfig

from app_logger.logger import logging

pd.set_option('display.max_columns', None)

st.title("Credit Card High-Risk Prediction")

# Train or Predict controls
if st.button("Train Model"):
    st.info("Starting training pipeline...")
    # Data ingestion
    data_ingest = DataIngestion()
    train_df, test_df = data_ingest.load_data()
    # Data transformation
    data_pipeline = DataPipeline()
    X_train, X_test = data_pipeline.data_transformation(train_df, test_df)
    # Model training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train)
    st.success("Model training complete.")
    # Evaluate on test set
    y_test = X_test['Is high risk']
    X_test_feats = X_test.drop(columns=['Is high risk'], errors='ignore')
    model = joblib.load(ModelTrainerConfig().trained_model_file_path)
    preds = model.predict(X_test_feats)
    acc = accuracy_score(y_test, preds)
    st.metric("Test set accuracy", f"{acc:.2f}")
    # st.write("First 10 predictions:")
    # st.write(pd.DataFrame({"Actual": y_test.values[:10], "Predicted": preds[:10]}))

if st.button("Predict"):
    st.info("Loading pipeline and model for inference...")
    # Load saved pipeline and model
    pipeline = pickle.load(open(DataTransformationConfig().transformation_obj_file_path, 'rb'))
    model = joblib.load(ModelTrainerConfig().trained_model_file_path)
    # Default test set inference
    st.subheader("Default Test Set Predictions")
    df_test = pd.read_csv("artifacts/data/test.csv")
    X_test_trans = pipeline.transform(df_test)
    X_test = X_test_trans.drop(columns=["Is high risk"], errors='ignore')
    y_test = X_test_trans.get("Is high risk")
    preds = model.predict(X_test)
    if y_test is not None:
        acc = accuracy_score(y_test, preds)
        st.metric("Test set accuracy", f"{acc:.2f}")
    st.write(pd.DataFrame({"Actual": y_test.values[:10] if y_test is not None else None, "Predicted": preds[:10]}))
    # # Custom upload
    # st.subheader("Upload a CSV for prediction")
    # uploaded = st.file_uploader("Choose a CSV file", type=["csv"] )
    # if uploaded:
    #     df_new = pd.read_csv(uploaded)
    #     st.write("Raw Input:")
    #     st.write(df_new.head())
    #     X_new_trans = pipeline.transform(df_new)
    #     preds_new = model.predict(X_new_trans)
    #     df_new["Prediction"] = preds_new
    #     st.subheader("Predictions on your data")
    #     st.write(df_new)

# st.write("\n---\nBuilt with Streamlit")
