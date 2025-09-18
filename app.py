# ===============================================
# RepoMind - BE Lab Project
# AI-powered GitHub Repository Analyzer
# Author: Your Name
# ===============================================

import os
import git
import streamlit as st
import google.generativeai as genai
from git.exc import GitCommandError

# --- Page Configuration ---
st.set_page_config(
    page_title="RepoMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Header ---
st.title("🤖 RepoMind AI: Repository Analyzer")
st.markdown("Get a quick, AI-powered summary of any public GitHub repository just by providing its URL. Understand its purpose, main features, and recent activity in seconds.")

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "",
        type="password",
        help="Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)"
    )
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key configured!", icon="✅")

# --- Core Functions ---
@st.cache_data(show_spinner=False)
def clone_and_extract(repo_url, max_commits=50):
    """
    Clones a repository to a local path and extracts commit messages.
    Uses st.cache_data to avoid re-cloning the same repo.
    """
    local_path = os.path.join("/tmp", repo_url.split("/")[-1].replace(".git", ""))

    try:
        if not os.path.exists(local_path):
            with st.spinner(f"Cloning repository: {repo_url}..."):
                git.Repo.clone_from(repo_url, local_path)
        else:
            # If repo exists, pull the latest changes
            with st.spinner("Fetching latest updates..."):
                repo = git.Repo(local_path)
                repo.remotes.origin.pull()
        
        repo = git.Repo(local_path)
        commits = []
        for i, commit in enumerate(repo.iter_commits()):
            if i >= max_commits:
                break
            commits.append(f"- {commit.message.strip()}")
        return commits, None  # Return commits and no error
    except GitCommandError as e:
        error_message = f"Error cloning or accessing repository: {e}. Please check if the URL is correct and the repository is public."
        return None, error_message
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        return None, error_message

@st.cache_data
def summarize_repo(messages, repo_url):
    """
    Summarizes a list of commit messages using the RepoMind LLM.
    """
    text = "\n".join(messages)
    prompt = f"""
    You are RepoMind, an expert AI software engineering analyst.
    Your task is to provide a high-level summary of a GitHub repository based on its recent commit history.

    Analyze the following commit messages from the repository at {repo_url}:
    --- COMMIT HISTORY ---
    {text}
    --- END OF HISTORY ---

    Based on this history, please provide a summary in markdown format that includes:
    1.  **Project Purpose & Main Features:** What is this project likely about? What does it do?
    2.  **Key Themes in Recent Activity:** What are the developers focusing on recently? (e.g., bug fixes, new features, refactoring, documentation).
    3.  **Overall Impression:** Give a brief overall impression of the project's health and direction based on the commits.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text, None # Return summary and no error
    except Exception as e:
        error_message = f"Failed to generate summary with AI: {e}. Check your API key and network."
        return None, error_message

# --- Main App Interface ---
st.subheader("Analyze a Repository")
repo_url = st.text_input(
    "Enter a public GitHub repository URL",
    "https://github.com/streamlit/streamlit.git",
    placeholder="e.g., https://github.com/tensorflow/tensorflow.git"
)

max_commits_to_analyze = st.slider(
    "Number of recent commits to analyze",
    min_value=20,
    max_value=200,
    value=50,
    step=10
)

if st.button("🚀 Analyze Repository", type="primary"):
    if not api_key:
        st.warning("Please enter your  API Key in the sidebar to proceed.", icon="⚠️")
    elif not repo_url:
        st.warning("Please enter a GitHub repository URL.", icon="🔗")
    else:
        # Step 1: Clone and Extract
        commit_messages, error = clone_and_extract(repo_url, max_commits_to_analyze)

        if error:
            st.error(error)
        else:
            st.success(f"Successfully collected {len(commit_messages)} commit messages.", icon="📂")
            
            # Display a preview of commits in an expander
            with st.expander("View Raw Commit Messages"):
                st.code("\n".join(commit_messages), language="text")

            # Step 2: Summarize with AI
            with st.spinner("🤖 RepoMind is analyzing the commits... Please wait."):
                summary, error = summarize_repo(commit_messages, repo_url)

            if error:
                st.error(error)
            else:
                st.subheader("📌 AI-Generated Repository Summary")
                st.markdown(summary)