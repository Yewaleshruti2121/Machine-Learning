import streamlit as st
import pickle

# Load the saved model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit page setup
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Spam Message Classifier")
st.markdown("Enter a message below to check if it's **Spam** or **Ham (Not Spam)**.")

# User input
input_message = st.text_area("Type your message here:")

# Predict button
if st.button("Predict"):
    # Vectorize the input text
    transformed_input = vectorizer.transform([input_message])

    # Predict using the trained model
    prediction = model.predict(transformed_input)[0]

    # Show result
    if prediction == 1:
        st.error("❌ This is a SPAM message!")
    else:
        st.success("✅ This is a HAM (not spam) message.")
