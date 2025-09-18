# RepoMind AI 🤖 - GitHub Repository Analyzer

RepoMind is a web application built with Streamlit and powered by Google's Gemini AI. It clones any public GitHub repository, analyzes its recent commit history, and provides a high-level technical summary.

This tool is perfect for developers, managers, or students who want to quickly understand the purpose, key features, and recent activity of a GitHub project without reading through all the code.



## Features

-   **Simple Web Interface**: Just paste a GitHub repository URL and click "Analyze".
-   **AI-Powered Summaries**: Uses the `gemini-1.5-flash` model to generate concise and insightful summaries from commit messages.
-   **Adjustable Analysis Depth**: Choose how many recent commits to analyze.
-   **Secure API Key Handling**: Uses Streamlit's password input to keep your Gemini API key safe.

## How to Run

**1. Prerequisites**
-   Python 3.8+
-   A Google Gemini API Key. You can get one from [Google AI Studio](https://makersuite.google.com/app/apikey).

**2. Clone this repository**
```bash
git clone <your-repo-link>
cd repomind_streamlit
```

**3. Install Dependencies**
Create a virtual environment (recommended) and install the required packages.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

**4. Run the Streamlit App**
From your terminal, run the following command:
```bash
streamlit run app.py
```
The application will open in your web browser. Enter your Gemini API key in the sidebar, provide a GitHub URL, and start analyzing!