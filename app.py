# ==================================================================================
# RepoMind AI - v3.5 (Final)
#
# Changelog:
# - Moved Key Statistics into the AI Summary tab for a consolidated view.
# - Added an automatic Project Preview feature to the Visualizations tab.
# - The app now searches for and displays a representative image from the repo.
# ==================================================================================

import os
import git
import json
import requests
import streamlit as st
import tempfile
import shutil
import time
import stat
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple, Generator, List

# ===============================================
# PAGE CONFIGURATION & STYLING
# ===============================================

st.set_page_config(
    page_title="RepoMind AI - Analyze & Chat",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced CSS for a professional look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #F0F2F6;
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        background: -webkit-linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.1s ease-in-out, box-shadow 0.1s ease-in-out;
        border: none;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #FFFFFF;
        color: #4F4F4F;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        border-color: #6a11cb;
        color: #6a11cb;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)


# ===============================================
# ANALYSIS & HELPER FUNCTIONS
# ===============================================

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def get_dir_size(path_str: str) -> str:
    total_size = 0
    start_path = Path(path_str)
    for dirpath, dirnames, filenames in os.walk(start_path):
        if '.git' in dirpath: continue
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp): total_size += os.path.getsize(fp)
    if total_size < 1024: return f"{total_size} Bytes"
    if total_size < 1024**2: return f"{total_size/1024:.2f} KB"
    if total_size < 1024**3: return f"{total_size/1024**2:.2f} MB"
    return f"{total_size/1024**3:.2f} GB"

def get_file_tree(repo_path: str, max_items=25) -> str:
    tree = []
    item_count = 0
    for root, dirs, files in os.walk(repo_path):
        if '.git' in root: continue
        level = root.replace(repo_path, '').count(os.sep)
        indent = ' ' * 4 * level
        if item_count < max_items:
            tree.append(f"{indent}📂 {os.path.basename(root)}/")
            item_count += 1
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if item_count < max_items:
                tree.append(f"{sub_indent}📄 {f}")
                item_count += 1
            else: break
        if item_count >= max_items:
            tree.append("...")
            break
    return "\n".join(tree)

# --- NEW: Function to find a preview image in the repository ---
def find_preview_image(repo_path: str) -> Optional[bytes]:
    """Searches for a potential preview image and returns its bytes if found."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif']
    search_keywords = ['preview', 'screenshot', 'demo', 'cover', 'hero', 'banner']
    
    candidate_files = []
    for root, _, files in os.walk(repo_path):
        if '.git' in root: continue
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                candidate_files.append(os.path.join(root, file))

    # Prioritize files with keywords in their name
    for keyword in search_keywords:
        for f in candidate_files:
            if keyword in f.lower():
                with open(f, 'rb') as img_file:
                    return img_file.read()
    
    # If no keyword match, return the first image found (if any)
    if candidate_files:
        with open(candidate_files[0], 'rb') as img_file:
            return img_file.read()
            
    return None

def find_and_parse_dependencies(repo_path: str) -> Dict[str, List[str]]:
    dependencies = {}
    req_file = os.path.join(repo_path, 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r', errors='ignore') as f:
            lines = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
            dependencies['Python (requirements.txt)'] = lines[:15]
    package_file = os.path.join(repo_path, 'package.json')
    if os.path.exists(package_file):
        with open(package_file, 'r', errors='ignore') as f:
            try:
                data = json.load(f)
                deps = list(data.get('dependencies', {}).keys())
                dev_deps = list(data.get('devDependencies', {}).keys())
                dependencies['JavaScript (package.json)'] = (deps + dev_deps)[:15]
            except json.JSONDecodeError: pass
    return dependencies

@st.cache_data(show_spinner=False, ttl=3600)
def clone_and_analyze_repository(repo_url: str) -> Tuple[Optional[Dict], Optional[str]]:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    temp_dir = Path(tempfile.gettempdir()) / f"repomind_{repo_name}_{datetime.now().timestamp()}"
    try:
        repo = git.Repo.clone_from(repo_url, str(temp_dir), depth=500)
        commits = list(repo.iter_commits(max_count=500))
        contributors = len({c.author.email for c in commits})
        latest_commit_date = datetime.fromtimestamp(commits[0].committed_date).strftime('%d %b %Y') if commits else "N/A"
        repo_size = get_dir_size(str(temp_dir))
        file_tree_summary = get_file_tree(str(temp_dir))
        preview_image = find_preview_image(str(temp_dir)) # Find preview image

        repo.close()
        
        return {
            'repo_name': repo_name, 'commit_patterns': analyze_commit_patterns(commits),
            'languages': detect_programming_languages(get_file_extension_stats(str(temp_dir))),
            'readme': extract_readme_content(str(temp_dir)), 'dependencies': find_and_parse_dependencies(str(temp_dir)),
            'total_files': sum(get_file_extension_stats(str(temp_dir)).values()), 'contributors': contributors,
            'latest_commit_date': latest_commit_date, 'repo_size': repo_size, 'file_tree': file_tree_summary,
            'preview_image': preview_image
        }, None
    except Exception as e:
        return None, f"❌ **Error cloning or analyzing repository:** {e}"
    finally:
        if temp_dir.exists():
            for i in range(3):
                try:
                    shutil.rmtree(temp_dir, onerror=remove_readonly)
                    break
                except PermissionError: time.sleep(1)

def analyze_commit_patterns(commits: list) -> Dict:
    commit_months = defaultdict(int)
    for c in commits: commit_months[datetime.fromtimestamp(c.committed_date).strftime('%Y-%m')] += 1
    return dict(sorted(commit_months.items())[-12:])

def get_file_extension_stats(repo_path: str) -> Dict[str, int]:
    extensions = defaultdict(int)
    for root, _, files in os.walk(repo_path):
        if '.git' in root: continue
        for file in files:
            ext = Path(file).suffix.lower()
            if ext: extensions[ext] += 1
    return dict(extensions)

def detect_programming_languages(extensions: Dict[str, int]) -> Dict[str, int]:
    lang_map = {'.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.java': 'Java', '.cpp': 'C++', '.c': 'C', '.cs': 'C#', '.go': 'Go', '.rs': 'Rust', '.html': 'HTML', '.css': 'CSS', '.md': 'Markdown'}
    languages = defaultdict(int)
    for ext, count in extensions.items():
        if lang := lang_map.get(ext): languages[lang] += count
    return dict(languages)

def extract_readme_content(repo_path: str) -> Optional[str]:
    for name in ['README.md', 'readme.md', 'README.rst']:
        path = os.path.join(repo_path, name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f: return f.read()
    return "No README file found."

def generate_ai_summary(analysis_data: Dict, endpoint: str, model: str) -> Optional[str]:
    context = {k: v for k, v in analysis_data.items() if k not in ['readme', 'commit_patterns', 'preview_image']}
    prompt = f"Analyze the following Git repository data and provide a concise, expert summary in Markdown:\n\n{json.dumps(context)}\n\nGenerate:\n1. **Project Purpose:** A one-sentence summary.\n2. **Key Technologies:** A bulleted list of tech.\n3. **Overall Impression:** A short paragraph on its purpose and audience."
    try:
        res = requests.post(f"{endpoint.rstrip('/')}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}, timeout=300)
        res.raise_for_status()
        return res.json().get('message', {}).get('content')
    except requests.RequestException as e:
        st.error(f"Failed to connect to Ollama: {e}")
        return None

def ollama_chatbot_stream(question: str, context: Dict, endpoint: str, model: str) -> Generator[str, None, None]:
    prompt = f"You are an expert AI assistant. Use the repository context below to answer the user's question. If the context is insufficient, say so.\n\n**CONTEXT:**\n---\n{json.dumps(context)}\n---\n\n**QUESTION:** {question}"
    try:
        res = requests.post(f"{endpoint.rstrip('/')}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True}, timeout=180, stream=True)
        res.raise_for_status()
        for line in res.iter_lines():
            if line: yield json.loads(line).get('message', {}).get('content', '')
    except requests.RequestException:
        yield "Error connecting to the Ollama server."

# ===============================================
# VISUALIZATION FUNCTIONS
# ===============================================

def create_language_chart(languages: Dict) -> go.Figure:
    if not languages: return go.Figure()
    fig = go.Figure(data=[go.Pie(labels=list(languages.keys()), values=list(languages.values()), hole=0.5, marker_colors=['#6a11cb', '#2575fc', '#8a2be2', '#00BFFF', '#7B68EE'])])
    fig.update_layout(title_text="<b>Language Distribution</b>", showlegend=True, height=450, font=dict(family="Inter, sans-serif"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_commit_timeline(commit_months: Dict) -> go.Figure:
    if not commit_months: return go.Figure()
    fig = go.Figure(data=[go.Scatter(x=list(commit_months.keys()), y=list(commit_months.values()), mode='lines+markers', line=dict(color='#2575fc', width=3), marker=dict(color='#6a11cb', size=8))])
    fig.update_layout(title_text="<b>Commit Activity</b> (Last 12 Mo.)", height=450, xaxis_title="", yaxis_title="", font=dict(family="Inter, sans-serif"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# ===============================================
# UI RENDERING FUNCTIONS
# ===============================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Connect to your remote Ollama model running on Google Colab.")
        ollama_endpoint = st.text_input("Ollama Endpoint (ngrok URL)", value="<PASTE_YOUR_NGROK_URL_HERE>", help="Paste public ngrok URL from Colab.")
        ollama_model = st.text_input("Ollama Model", value="llama3:8b", help="For better accuracy, try llama3:8b or mistral.")
        st.divider()
        st.markdown("#### How It Works")
        st.markdown("1. Paste your `ngrok` URL.\n2. Choose an example or enter a URL.\n3. Click 'Analyze'.\n4. Chat with the AI!")
        return ollama_endpoint, ollama_model

def render_input_section():
    st.markdown("##### ✨ Select an Example Repository")
    example_repos = {"Streamlit": "https://github.com/streamlit/streamlit", "VS Code": "https://github.com/microsoft/vscode", "React": "https://github.com/facebook/react", " Python": "https://github.com/vinta/awesome-python"}
    cols = st.columns(len(example_repos))
    for i, (name, url) in enumerate(example_repos.items()):
        if cols[i].button(name, use_container_width=True):
            st.session_state.repo_url = url
    st.text_input("Or enter a public GitHub Repository URL:", key="repo_url", placeholder="https://github.com/user/repo")

# --- MODIFIED: This function now includes the key statistics ---
def render_summary_tab(data):
    st.subheader("🤖 AI Generated Summary")
    st.markdown(data.get('ai_summary', "No summary generated."), unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📈 Key Statistics")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total Files", data.get('total_files', 'N/A'), help="Total files (excluding .git).")
    stat_cols[1].metric("Repo Size", data.get('repo_size', 'N/A'), help="Total size (excluding .git).")
    stat_cols[2].metric("Contributors", data.get('contributors', 'N/A'), help="Unique authors in last 500 commits.")
    stat_cols[3].metric("Latest Commit", data.get('latest_commit_date', 'N/A'))

# --- MODIFIED: This function now includes the project preview image ---
def render_visualizations_tab(data):
    st.subheader("🖼️ Project Preview")
    if image_bytes := data.get('preview_image'):
        st.image(image_bytes, caption="A preview image found in the repository.", use_column_width=True)
    else:
        st.info("No preview image (e.g., screenshot.png) was found in the repository.")
    
    st.markdown("---")
    st.subheader("🎨 Visualizations")
    vis_cols = st.columns(2)
    with vis_cols[0]:
        st.plotly_chart(create_language_chart(data['languages']), use_container_width=True)
    with vis_cols[1]:
        st.plotly_chart(create_commit_timeline(data['commit_patterns']), use_container_width=True)

def render_code_details_tab(data):
    st.subheader("📂 Code Details")
    readme_tab, deps_tab = st.tabs(["📄 README", "📦 Dependencies"])
    with readme_tab:
        st.markdown(data.get('readme', "No README found."), unsafe_allow_html=True)
    with deps_tab:
        if data['dependencies']:
            for file, deps in data['dependencies'].items():
                st.markdown(f"**{file}:**")
                st.text('\n'.join(f"• {dep}" for dep in deps))
        else:
            st.info("No dependency files (like requirements.txt or package.json) found.")

def render_chat_tab(ollama_endpoint, ollama_model):
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the repository..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            chat_context = {
                "repo_name": st.session_state.analysis_data.get('repo_name'),
                "ai_summary": st.session_state.analysis_data.get('ai_summary'),
                "languages": st.session_state.analysis_data.get('languages'),
                "file_tree": st.session_state.analysis_data.get('file_tree'),
                "full_readme": st.session_state.analysis_data.get('readme', '')
            }
            full_response = st.write_stream(ollama_chatbot_stream(prompt, chat_context, ollama_endpoint, ollama_model))
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# ===============================================
# MAIN STREAMLIT APPLICATION
# ===============================================

def main():
    st.markdown('<h1 class="main-header">RepoMind AI</h1>', unsafe_allow_html=True)
    for key in ['analysis_data', 'analysis_error', 'repo_url', 'messages']:
        if key not in st.session_state:
            st.session_state[key] = [] if key == 'messages' else None
    
    ollama_endpoint, ollama_model = render_sidebar()
    render_input_section()

    if st.button("Analyze Repository", type="primary", use_container_width=True):
        if not st.session_state.repo_url:
            st.warning("Please enter a repository URL or select an example.")
        elif ollama_endpoint == "<PASTE_YOUR_NGROK_URL_HERE>" or not ollama_endpoint:
            st.error("Please update the 'Ollama Endpoint' in the sidebar.")
        else:
            with st.spinner("Performing deep analysis... This may take a moment."):
                analysis_data, error = clone_and_analyze_repository(st.session_state.repo_url)
                st.session_state.analysis_data = analysis_data if not error else None
                st.session_state.analysis_error = error
            if st.session_state.analysis_data:
                with st.spinner("🤖 Contacting Colab AI for a summary..."):
                    summary = generate_ai_summary(st.session_state.analysis_data, ollama_endpoint, ollama_model)
                    st.session_state.analysis_data['ai_summary'] = summary
                    if not summary: st.error("Could not generate AI summary.")

    st.markdown("---")

    if st.session_state.get('analysis_error'):
        st.error(st.session_state.analysis_error)
        
    if data := st.session_state.get('analysis_data'):
        # --- MODIFIED: Removed the "Key Statistics" tab ---
        tab_list = ["🤖 AI Summary & Stats", "🎨 Visualizations", "📂 Code Details", "💬 Chat with Repo AI"]
        summary_tab, viz_tab, code_tab, chat_tab = st.tabs(tab_list)

        with summary_tab:
            render_summary_tab(data)
        with viz_tab:
            render_visualizations_tab(data)
        with code_tab:
            render_code_details_tab(data)
        with chat_tab:
            render_chat_tab(ollama_endpoint, ollama_model)
    else:
        st.info("👋 Welcome to RepoMind AI!")

if __name__ == "__main__":
    main()
