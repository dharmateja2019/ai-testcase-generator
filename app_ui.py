import streamlit as st
from main import generate_test_cases
from utils import save_history
import json
import pandas as pd

st.set_page_config(page_title="AI Test Case Generator", layout="centered")

st.title("🤖 AI Test Case Generator")
st.write("Generate test cases from user stories using local LLM")

# 🔹 Model Selection (GLOBAL)
model_choice = st.selectbox(
    "Select Model",
    ["phi3:mini", "llama3.1:8b"]
)

# 🔹 File Upload Section
st.subheader("📂 Batch Processing")
uploaded_file = st.file_uploader("Upload user stories (.txt)", type=["txt"])

def read_user_stories(file):
    content = file.read().decode("utf-8")
    return content.split("\n")


def display_results(test_cases, raw_output, user_story):
    if test_cases:
        df = pd.DataFrame(test_cases)
        st.dataframe(df)
        save_history(user_story, test_cases)

        st.success("✅ Test cases generated!")

        # JSON
        st.subheader("📄 JSON Output")
        st.json(test_cases)

        # Table
        df = pd.DataFrame(test_cases)
        st.subheader("📊 Table View")
        st.dataframe(df)

        # Downloads
        st.download_button(
            "Download JSON",
            data=json.dumps(test_cases, indent=4),
            file_name="test_cases.json"
        )

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="test_cases.csv"
        )

    else:
        st.error("❌ Failed to generate structured test cases")
        st.text(raw_output)

if uploaded_file and st.button("Process File", key="batch_btn"):
    stories = read_user_stories(uploaded_file)

    for story in stories:
        if story.strip():
            raw_output, test_cases = generate_test_cases(story, model=model_choice)
            display_results(test_cases, raw_output, story)

# 🔹 Single Input Section
st.subheader("✍️ Single User Story")
user_story = st.text_area("Enter User Story")

if st.button("Generate Test Cases", key="single_btn"):
    if user_story.strip() == "":
        st.warning("Please enter a user story")
    else:
        with st.spinner("🔍 Analyzing user story and generating test cases..."):
            raw_output, test_cases = generate_test_cases(user_story, model=model_choice)
            display_results(test_cases, raw_output, user_story)


# 🔹 History Section (Independent)
st.subheader("🕘 History")

if st.checkbox("Show Last 5 Runs"):
    try:
        with open("history.json") as f:
            history = json.load(f)

        for item in reversed(history[-5:]):
            st.write(item["timestamp"])
            st.write(item["user_story"])
            st.json(item["output"])
            st.divider()

    except:
        st.write("No history found")
st.divider()
if st.button("Clear"):
    st.session_state.clear()