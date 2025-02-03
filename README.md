# House Price Prediction
## DEMO
https://github.com/user-attachments/assets/3910d3a8-9fb7-4f84-92d4-16a67a99e055

This project predicts house prices using a trained deep learning model. The application is built using Streamlit for easy interaction with the model.

## Dataset link
https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction

## Features
- Train a house price prediction model using deep learning.
- Perform inference on new house data.
- Scale and encode input features.
- Visualize actual vs. predicted prices.
- Provide a user-friendly UI for manual input or test dataset selection.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install streamlit pandas numpy joblib tensorflow matplotlib seaborn scikit-learn
```

## Project Structure
```
.
├── house_price_prediction.ipynb  # Jupyter Notebook for model training
├── infer_app.py                  # Streamlit application for inference
├── tf_housing_model.h5           # Trained TensorFlow model
├── scaler_X.pkl                  # Scaler for input features
├── scaler_y.pkl                  # Scaler for target variable
├── feature_order.json             # Order of features used in training
├── Housing.csv                    # Dataset for inference
```

## Running the Inference Application
To launch the Streamlit web application, use the following command:

```bash
streamlit run infer_app.py
```

## How It Works
1. **Manual Input Mode**
   - Enter property details manually.
   - Click on the `Predict Price` button to get the estimated house price.

2. **Test Set Mode**
   - Select a test row from the dataset.
   - View the actual price of the selected house.
   - Click `Predict Price from Test Row` to compare actual vs. predicted price.
   - View the error margin and visualization.

## Model Training (Jupyter Notebook)
- The `house_price_prediction.ipynb` notebook contains the training pipeline.
- The dataset is preprocessed, categorical variables are encoded, and numerical features are scaled.
- The model is trained and saved as `tf_housing_model.h5`.

## Notes
- Ensure that the required files (`tf_housing_model.h5`, `scaler_X.pkl`, `scaler_y.pkl`, `feature_order.json`, `Housing.csv`) exist before running the inference app.
- If files are missing, train the model using the Jupyter notebook.
