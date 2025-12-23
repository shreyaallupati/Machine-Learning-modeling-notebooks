
import streamlit as st
import pandas as pd
import pickle

# Function to load the model (cached with st.cache)
@st.cache(allow_output_mutation=True)
def load_model():
    with open('final_xgb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to make predictions (cached with st.cache)
@st.cache(allow_output_mutation=True)
def predict_churn(model, data):
    predictions = model.predict(data)
    predicted_labels = ['Not Churn' if pred == 0 else 'Churn' for pred in predictions]
    return predicted_labels

# Main function to run the Streamlit app
def main():
    st.title('Customer Churn Prediction')
    st.sidebar.title('User Input')

    # Load the model
    model = load_model()

    # Create example data
    example_data = pd.DataFrame({
        'gender': [1, 0, 0, 0, 0],
        'SeniorCitizen': [0, 0, 0, 0, 0],
        'Partner': [0, 0, 0, 1, 1],
        'Dependents': [0, 0, 0, 0, 1],
        'PhoneService': [1, 0, 1, 1, 1],
        'MultipleLines': [0, 0, 0, 2, 2],
        'InternetService': [1, 0, 1, 1, 0],
        'OnlineSecurity': [0, 0, 0, 2, 2],
        'OnlineBackup': [0, 0, 1, 2, 2],
        'DeviceProtection': [0, 0, 0, 0, 2],
        'TechSupport': [0, 0, 0, 2, 2],
        'StreamingTV': [0, 1, 0, 0, 0],
        'StreamingMovies': [0, 1, 0, 0, 0],
        'Contract': [2, 0, 0, 1, 2],
        'PaperlessBilling': [0, 1, 0, 0, 0],
        'PaymentMethod': [1, 1, 1, 0, 0],
        'MonthlyCharges': [90.407734, 58.273891, 74.379767, 108.55, 64.35],
        'TotalCharges': [707.535237, 3264.466697, 1146.937795, 5610.7, 1558.65],
        'tenure_group': [0, 4, 1, 4, 2]
    })

    # Display example data in sidebar
    st.sidebar.write('Example Input Data:')
    st.sidebar.write(example_data)

    # Allow user input for custom data
    st.sidebar.write('Custom Input Data:')
    custom_data = {}
    for col in example_data.columns:
        custom_data[col] = st.sidebar.selectbox(col, example_data[col])

    # Prepare data for prediction
    input_data = pd.DataFrame([custom_data])

    # Make predictions using cached function
    predictions = predict_churn(model, input_data)

    # Display predictions
    st.subheader('Prediction')
    st.write(predictions[0])

    # Display detailed output
    st.subheader('Detailed Output')
    st.write(input_data)

# Run the main function to start the app
if __name__ == '__main__':
    main()
