import pandas as pd
import cohere
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    st.error("Cohere API key is missing or invalid.")
    st.stop()

cohere_client = cohere.Client(api_key)

def load_fitness_data(file_path):
    try:
        data_frame = pd.read_csv(file_path)

        return data_frame
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

fitness_data = load_fitness_data('megaGymDataset.csv')

def get_user_preferences():
    try:
        fitness_goal = st.selectbox("What's your primary fitness goal?",
                                    ["Weight Loss", "Build Muscle", "Endurance", "General Fitness"])
        fitness_level = st.radio("What's your experience level?",
                                 ["Beginner", "Intermediate", "Advanced"])
        has_restrictions = st.checkbox("Any injuries or limitations?")
        return {"goal": fitness_goal, "experience": fitness_level, "restrictions": has_restrictions}
    except Exception as e:
        st.error(f"Error gathering user preferences: {e}")
        st.stop()

def handle_query(user_query, fitness_data, preferences):
    try:
        query_prompt = create_fitness_prompt(user_query, fitness_data, preferences)

        # Make API call
        response = cohere_client.chat(message=query_prompt)

        # Check and return response
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Invalid response format or empty response."
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return "Sorry, something went wrong."

# --- Helper Functions ---
def create_fitness_prompt(query, data, preferences):
    # Construct a more specific prompt based on user preferences
    return (f"User Query: {query}\n"
            f"User Preferences: {preferences}\n"
            f"Provide a detailed fitness response based on the following data: {data.head().to_dict()}")

# --- Streamlit UI ---
st.title("Fitness Plan Generatation")

user_prefs = get_user_preferences()

user_query_input = st.text_input("Ask me about workouts or fitness...")

if st.button("Submit"):
    response_text = handle_query(user_query_input, fitness_data, user_prefs)
    st.write("Response:", response_text)
