import joblib
import streamlit as st
import requests
from io import BytesIO

# Load trained model
def load_model():
    url = 'https://huggingface.co/ioakowuah/classificationmodelnew/resolve/main/RandomForestClassifier_model%20(10).pkl'
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

model = load_model()

# Main app
def main():
    st.title('💼 Will the client subscribe to a term deposit?')

    # Encoding map for poutcome and month
    month_map = {
        'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
        'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
    }

    poutcome_map = {
        'unknown': 0, 'failure': 1, 'other': 2, 'success': 3
    }

    # User inputs (only selected features)
    age = st.number_input('Client Age', min_value=18, max_value=100)
    duration = st.number_input('Last Contact Duration (in seconds)', min_value=0)
    day = st.number_input('Day of Last Contact (1-31)', min_value=1, max_value=31)
    month = st.selectbox('Month of Last Contact', list(month_map.keys()))
    pdays = st.number_input('Days Since Previous Campaign Contact', min_value=0)
    poutcome = st.selectbox('Outcome of Previous Campaign', list(poutcome_map.keys()))

    # Encode inputs
    month_encoded = month_map[month]
    poutcome_encoded = poutcome_map[poutcome]

    # Prediction
    if st.button('Predict'):
        try:
            input_data = [[
                duration, age, day, month_encoded, pdays, poutcome_encoded
            ]]
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.success("✅ Client **will subscribe** to a term deposit.")
            else:
                st.warning("❌ Client **will not subscribe** to a term deposit.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Run the app
if __name__ == '__main__':
    main()
