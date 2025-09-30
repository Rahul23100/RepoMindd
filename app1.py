# ===============================================
# RepoMind AI - Enhanced Repository Analyzer
# Complete Single-File Application
# Features: AI Analysis, Code Stats, Language Detection, Contributors, Real-time Chat
# ===============================================

import os
import re
import git
import json
import time
import base64
import requests
import pandas as pd
import streamlit as st
# removed google.generativeai import to avoid dependency; using OpenAI HTTP API via requests
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from git.exc import GitCommandError
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional

# ===============================================
# PAGE CONFIGURATION
# ===============================================

st.set_page_config(
    page_title="RepoMind AI - Advanced Repository Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/repomind',
        'Report a bug': "https://github.com/yourusername/repomind/issues",
        'About': "# RepoMind AI\nAdvanced AI-powered repository analyzer for understanding GitHub projects instantly."
    }
)

# ===============================================
# CUSTOM CSS STYLING
# ===============================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main Theme */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.15);
    }
    
    /* Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Success Alert */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 10px;
        color: #1a1a1a;
        font-weight: 500;
        margin: 10px 0;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Spinner Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def get_file_extension_stats(repo_path: str) -> Dict[str, int]:
    """Analyze file types in the repository."""
    extensions = defaultdict(int)
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if not file.startswith('.'):
                ext = Path(file).suffix.lower()
                if ext:
                    extensions[ext] += 1
    return dict(extensions)

def detect_programming_languages(extensions: Dict[str, int]) -> Dict[str, int]:
    """Detect programming languages based on file extensions."""
    language_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.cs': 'C#',
        '.rb': 'Ruby',
        '.go': 'Go',
        '.rs': 'Rust',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.php': 'PHP',
        '.r': 'R',
        '.m': 'MATLAB',
        '.scala': 'Scala',
        '.sh': 'Shell',
        '.html': 'HTML',
        '.css': 'CSS',
        '.vue': 'Vue',
        '.jsx': 'React',
        '.tsx': 'TypeScript React'
    }
    
    languages = defaultdict(int)
    for ext, count in extensions.items():
        if ext in language_map:
            languages[language_map[ext]] += count
    
    return dict(languages)

def analyze_commit_patterns(repo) -> Dict:
    """Analyze commit patterns and contributor statistics."""
    commits = list(repo.iter_commits(max_count=500))
    
    # Time-based analysis
    commit_hours = Counter()
    commit_days = Counter()
    commit_months = defaultdict(int)
    
    # Contributor analysis
    contributors = Counter()
    contributor_commits = defaultdict(list)
    
    for commit in commits:
        dt = datetime.fromtimestamp(commit.committed_date)
        commit_hours[dt.hour] += 1
        commit_days[dt.strftime('%A')] += 1
        commit_months[dt.strftime('%Y-%m')] += 1
        
        author = commit.author.name if commit.author else "Unknown"
        contributors[author] += 1
        contributor_commits[author].append({
            'message': commit.message[:100],
            'date': dt.strftime('%Y-%m-%d'),
            'hash': commit.hexsha[:7]
        })
    
    # Calculate velocity (commits per day over last 30 days)
    recent_commits = [c for c in commits if 
                      datetime.fromtimestamp(c.committed_date) > 
                      datetime.now() - timedelta(days=30)]
    velocity = len(recent_commits) / 30 if recent_commits else 0
    
    return {
        'total_commits': len(commits),
        'commit_hours': dict(commit_hours),
        'commit_days': dict(commit_days),
        'commit_months': dict(sorted(commit_months.items())[-12:]),
        'contributors': dict(contributors.most_common(10)),
        'contributor_details': dict(list(contributor_commits.items())[:5]),
        'velocity': round(velocity, 2),
        'most_active_hour': max(commit_hours, key=commit_hours.get) if commit_hours else None,
        'most_active_day': max(commit_days, key=commit_days.get) if commit_days else None
    }

def extract_readme_content(repo_path: str) -> Optional[str]:
    """Extract README content if available."""
    readme_files = ['README.md', 'readme.md', 'README.rst', 'README.txt']
    for readme in readme_files:
        readme_path = os.path.join(repo_path, readme)
        if os.path.exists(readme_path):
            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    return f.read()[:5000] # Limit README size
            except:
                continue
    return None

def calculate_code_metrics(repo_path: str) -> Dict:
    """Calculate basic code metrics."""
    total_lines = 0
    total_files = 0
    largest_file = {"name": "", "lines": 0}
    
    code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts', '.rb', '.php'}
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if Path(file).suffix.lower() in code_extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        total_files += 1
                        if lines > largest_file["lines"]:
                            largest_file = {
                                "name": os.path.relpath(file_path, repo_path),
                                "lines": lines
                            }
                except:
                    continue
    
    return {
        'total_lines': total_lines,
        'total_files': total_files,
        'avg_lines_per_file': round(total_lines / total_files) if total_files > 0 else 0,
        'largest_file': largest_file
    }

# ===============================================
# Helper: OpenAI HTTP API Caller (using requests)
# ===============================================
def call_openai_chat(api_key: str, messages: List[Dict], model: str = "gpt-4o-mini", max_tokens: int = 800, temperature: float = 0.2) -> Tuple[Optional[str], Optional[str]]:
    """
    Call OpenAI Chat Completions endpoint using requests.
    Returns (text, error_message).
    """
    if not api_key:
        return None, "No API key provided."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            try:
                err = resp.json()
                return None, f"OpenAI API error: {resp.status_code} - {err.get('error', err)}"
            except Exception:
                return None, f"OpenAI API error: {resp.status_code} - {resp.text}"
        data = resp.json()
        # Compose text from choices
        text = ""
        for choice in data.get("choices", []):
            part = choice.get("message", {}).get("content", "")
            text += part
        return text.strip(), None
    except Exception as e:
        return None, f"Request error: {str(e)}"

# ===============================================
# CORE REPOSITORY ANALYSIS FUNCTIONS
# ===============================================

@st.cache_data(show_spinner=False, ttl=3600)
def clone_and_analyze_repository(repo_url: str, max_commits: int = 100) -> Tuple[Dict, Optional[str]]:
    """
    Clone repository and perform comprehensive analysis.
    Returns analysis results and error message if any.
    """
    # Clean repo URL
    repo_url = repo_url.strip().rstrip('/')
    if not repo_url.endswith('.git'):
        repo_url += '.git'
    
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    local_path = os.path.join("/tmp", f"repomind_{repo_name}_{hash(repo_url)}")
    
    try:
        # Clone or update repository
        if not os.path.exists(local_path):
            with st.spinner(f"🔄 Cloning repository: {repo_name}..."):
                repo = git.Repo.clone_from(repo_url, local_path, depth=max_commits)
        else:
            with st.spinner("📥 Fetching latest updates..."):
                repo = git.Repo(local_path)
                repo.remotes.origin.pull()
        
        # Extract commit messages
        commits = []
        commit_data = []
        for i, commit in enumerate(repo.iter_commits()):
            if i >= max_commits:
                break
            commits.append(commit.message.strip())
            commit_data.append({
                'hash': commit.hexsha[:7],
                'author': commit.author.name if commit.author else "Unknown",
                'date': datetime.fromtimestamp(commit.committed_date),
                'message': commit.message.strip()[:100]
            })
        
        # Perform comprehensive analysis
        with st.spinner("🔍 Analyzing repository structure..."):
            file_stats = get_file_extension_stats(local_path)
            languages = detect_programming_languages(file_stats)
            commit_patterns = analyze_commit_patterns(repo)
            readme = extract_readme_content(local_path)
            code_metrics = calculate_code_metrics(local_path)
        
        # Get repository size
        repo_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(local_path)
                        for filename in filenames) / (1024 * 1024)  # MB
        
        analysis = {
            'repo_name': repo_name,
            'repo_url': repo_url.replace('.git', ''),
            'local_path': local_path,
            'commits': commits,
            'commit_data': commit_data,
            'file_stats': file_stats,
            'languages': languages,
            'commit_patterns': commit_patterns,
            'readme': readme,
            'code_metrics': code_metrics,
            'repo_size': round(repo_size, 2),
            'total_files': sum(file_stats.values()),
            'branch_count': len(list(repo.branches)),
            'has_issues': 'issues' in repo_url.lower(),
            'last_update': commit_data[0]['date'].strftime('%Y-%m-%d') if commit_data else "Unknown"
        }
        
        return analysis, None
        
    except GitCommandError as e:
        return None, f"❌ Git Error: {str(e)}. Please ensure the repository is public and the URL is correct."
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"

@st.cache_data(ttl=3600)
def generate_ai_summary(analysis: Dict, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate comprehensive AI summary using OpenAI chat completions.
    """
    # Prepare context for AI
    commits_text = "\n".join(analysis['commits'][:50])
    languages_text = ", ".join([f"{lang} ({count} files)" 
                                  for lang, count in list(analysis['languages'].items())[:5]])
    readme_snippet = analysis['readme'][:1000] if analysis['readme'] else "No README found"
    
    prompt_system = {
        "role": "system",
        "content": "You are RepoMind AI, an expert software repository analyst. Provide a concise, actionable markdown report with emojis where helpful."
    }
    prompt_user = {
        "role": "user",
        "content": f"""
REPOSITORY: {analysis['repo_name']}
URL: {analysis['repo_url']}

STATISTICS:
- Total Files: {analysis['total_files']}
- Repository Size: {analysis['repo_size']} MB
- Total Commits Analyzed: {len(analysis['commits'])}
- Contributors: {len(analysis['commit_patterns']['contributors'])}
- Code Lines: {analysis['code_metrics']['total_lines']}
- Last Updated: {analysis['last_update']}

LANGUAGES DETECTED:
{languages_text}

README EXCERPT:
{readme_snippet}

RECENT COMMITS:
{commits_text}

COMMIT VELOCITY: {analysis['commit_patterns']['velocity']} commits/day (last 30 days)

Please provide:
1. Project Purpose & Overview
2. Key Features & Capabilities (bullet list)
3. Technical Architecture (main tech & structure)
4. Development Activity & Health
5. Team & Community summary
6. Future Direction (suggested roadmap)
7. Recommendations for contributors/users
8. Potential Concerns / Red flags

Keep response in markdown, concise and actionable.
"""
    }
    messages = [prompt_system, prompt_user]
    text, err = call_openai_chat(api_key, messages, model="gpt-4o-mini", max_tokens=900, temperature=0.15)
    if err:
        return None, f"❌ AI Generation Error: {err}"
    return text, None

def generate_code_quality_report(analysis: Dict, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a code quality and best practices report using OpenAI.
    """
    prompt_system = {"role": "system", "content": "You are a senior software architect. Provide a concise code quality report with actionable recommendations."}
    prompt_user = {
        "role": "user",
        "content": f"""
As a senior software architect, analyze the code quality indicators for this repository: {analysis['repo_name']}

METRICS:
- Total Code Lines: {analysis['code_metrics']['total_lines']}
- Average Lines per File: {analysis['code_metrics']['avg_lines_per_file']}
- Largest File: {analysis['code_metrics']['largest_file']['name']} ({analysis['code_metrics']['largest_file']['lines']} lines)
- Languages: {', '.join(analysis['languages'].keys())}

COMMIT PATTERNS:
- Most Active Day: {analysis['commit_patterns']['most_active_day']}
- Most Active Hour: {analysis['commit_patterns']['most_active_hour']}:00
- Commit Velocity: {analysis['commit_patterns']['velocity']} commits/day

Recent commit messages:
{chr(10).join(analysis['commits'][:20])}

Please provide:
1. Code Organization assessment
2. Commit Quality assessment
3. Development Practices
4. Top 3 specific recommendations

Keep it concise and in markdown.
"""
    }
    messages = [prompt_system, prompt_user]
    text, err = call_openai_chat(api_key, messages, model="gpt-4o-mini", max_tokens=700, temperature=0.15)
    if err:
        return None, f"❌ Code quality analysis error: {err}"
    return text, None

def generate_chat_response(analysis: Dict, chat_history: List[Dict], user_question: str, api_key: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a conversational response based on the repository context and chat history using OpenAI.
    """
    # Prepare context for the AI
    readme_snippet = analysis['readme'][:2000] if analysis['readme'] else "No README found."
    languages_text = ", ".join(analysis['languages'].keys())
    file_list = "\n".join([f"- {ext} ({count} files)" for ext, count in list(analysis['file_stats'].items())[:10]])

    # Build the system + user + history messages
    system_msg = {
        "role": "system",
        "content": "You are RepoMind AI, an expert assistant helping to understand GitHub repositories. Answer concisely and cite repository context when relevant."
    }

    # Compose conversation history
    messages = [system_msg]
    # add repository context as a user message (so model can reference)
    repo_context = f"""
REPOSITORY: {analysis['repo_name']}
Purpose (README excerpt): {readme_snippet}

Core Technologies: {languages_text}

File Structure Snapshot:
{file_list}

Key Metrics:
- Total Files: {analysis['total_files']}
- Lines of Code: {analysis['code_metrics']['total_lines']}
- Contributors: {len(analysis['commit_patterns']['contributors'])}
"""
    messages.append({"role": "user", "content": repo_context})

    # Add prior chat history
    for msg in chat_history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})

    # Append current user question
    messages.append({"role": "user", "content": user_question})

    text, err = call_openai_chat(api_key, messages, model="gpt-4o-mini", max_tokens=600, temperature=0.18)
    if err:
        return None, f"❌ AI Chat Error: {err}"
    return text, None

# ===============================================
# VISUALIZATION FUNCTIONS
# ===============================================

def create_language_chart(languages: Dict) -> go.Figure:
    """Create an interactive pie chart of programming languages."""
    if not languages:
        return go.Figure()
    
    fig = go.Figure(data=[go.Pie(
        labels=list(languages.keys()),
        values=list(languages.values()),
        hole=0.3,
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='white', width=2)
        ),
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Files: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Programming Languages Distribution",
        showlegend=True,
        height=400,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_commit_timeline(commit_months: Dict) -> go.Figure:
    """Create a timeline chart of commit activity."""
    if not commit_months:
        return go.Figure()
    
    months = list(commit_months.keys())
    counts = list(commit_months.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=counts,
        mode='lines+markers',
        name='Commits',
        line=dict(
            color='rgb(102, 126, 234)',
            width=3,
            shape='spline'
        ),
        marker=dict(
            size=10,
            color='rgb(102, 126, 234)',
            line=dict(color='white', width=2)
        ),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>%{x}</b><br>Commits: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Commit Activity Timeline (Last 12 Months)",
        xaxis_title="Month",
        yaxis_title="Number of Commits",
        showlegend=False,
        height=350,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        hovermode='x unified'
    )
    
    return fig

def create_contributor_chart(contributors: Dict) -> go.Figure:
    """Create a bar chart of top contributors."""
    if not contributors:
        return go.Figure()
    
    names = list(contributors.keys())[:10]
    commits = list(contributors.values())[:10]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=commits,
        y=names,
        orientation='h',
        marker=dict(
            color=commits,
            colorscale='Viridis',
            line=dict(color='white', width=1)
        ),
        text=commits,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Commits: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Top Contributors",
        xaxis_title="Number of Commits",
        showlegend=False,
        height=400,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=150)
    )
    
    return fig

def create_activity_heatmap(commit_patterns: Dict) -> go.Figure:
    """Create a heatmap of commit activity by day and hour."""
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    
    # Create matrix for heatmap
    z = [[0 for _ in hours] for _ in days_order]
    
    # This requires a more detailed data structure from the analysis step.
    # For now, we'll create a simplified approximation.
    # A proper implementation would iterate through commits and fill a 2D array.
    for i, day in enumerate(days_order):
        for j, hour in enumerate(hours):
            # Simple approximation based on separate day/hour counts
            day_commits = commit_patterns['commit_days'].get(day, 0)
            hour_commits = commit_patterns['commit_hours'].get(hour, 0)
            # A more accurate model would require commit data with (day, hour) tuples.
            # This is an estimation for visualization.
            z[i][j] = (day_commits * hour_commits) / (commit_patterns['total_commits'] + 1)
            
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"{h:02d}:00" for h in hours],
        y=days_order,
        colorscale='Blues',
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Activity Score: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Commit Activity Heatmap",
        xaxis_title="Hour of Day",
        height=350,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ===============================================
# STREAMLIT UI COMPONENTS
# ===============================================

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">RepoMind AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2em; color: #666; margin-bottom: 30px;">'
        'Advanced AI-Powered GitHub Repository Analyzer</p>',
        unsafe_allow_html=True
    )

def display_metrics(analysis: Dict):
    """Display repository metrics in a grid."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Total Files",
            value=f"{analysis['total_files']:,}",
            help=f"{len(analysis['languages'])} languages detected"
        )
    
    with col2:
        st.metric(
            label="💾 Repository Size",
            value=f"{analysis['repo_size']} MB",
            help=f"{analysis['code_metrics']['total_lines']:,} lines of code"
        )
    
    with col3:
        st.metric(
            label="👥 Contributors",
            value=len(analysis['commit_patterns']['contributors']),
            help=f"Velocity: {analysis['commit_patterns']['velocity']} commits/day"
        )
    
    with col4:
        st.metric(
            label="📅 Last Updated",
            value=analysis['last_update'],
            help=f"{analysis['branch_count']} branches"
        )

def display_commit_history(commit_data: List[Dict]):
    """Display recent commits in a table."""
    if commit_data:
        df = pd.DataFrame(commit_data[:20])
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.markdown("### 📜 Recent Commits")
        st.dataframe(
            df[['hash', 'author', 'date', 'message']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "hash": st.column_config.TextColumn("Hash", width="small"),
                "author": st.column_config.TextColumn("Author", width="medium"),
                "date": st.column_config.TextColumn("Date", width="medium"),
                "message": st.column_config.TextColumn("Message", width="large")
            }
        )

def display_code_explorer(analysis: Dict):
    """Display code structure explorer."""
    with st.expander("🔍 **Code Structure Explorer**", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 File Type Distribution")
            top_extensions = dict(sorted(analysis['file_stats'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10])
            for ext, count in top_extensions.items():
                st.write(f"`{ext}`: {count} files")
        
        with col2:
            st.markdown("#### 📈 Code Metrics")
            st.write(f"**Total Lines:** {analysis['code_metrics']['total_lines']:,}")
            st.write(f"**Total Files:** {analysis['code_metrics']['total_files']:,}")
            st.write(f"**Avg Lines/File:** {analysis['code_metrics']['avg_lines_per_file']:,}")
            st.write(f"**Largest File:** `{analysis['code_metrics']['largest_file']['name']}`")
            st.write(f"**Lines in Largest:** {analysis['code_metrics']['largest_file']['lines']:,}")

# ===============================================
# MAIN APPLICATION
# ===============================================

def main():
    """Main application logic."""
    
    # Display header
    display_header()
    
    # Initialize session state variables
    if 'repo_url' not in st.session_state:
        st.session_state.repo_url = ""
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            " API Key",
            type="password",
            placeholder="Enter your OpenAI API key (sk-...)",
            help="Paste your OpenAI API key here (starts with sk-)."
        )
        
        if api_key:
            st.success("✅ API Key configured!")
        
        st.markdown("---")
        
        # Analysis settings
        st.markdown("## 🎛️ Analysis Settings")
        
        max_commits = st.slider(
            "Commits to Analyze",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Number of recent commits to analyze"
        )
        
        enable_visualizations = st.checkbox(
            "Enable Visualizations",
            value=True,
            help="Show interactive charts and graphs"
        )
        
        enable_code_quality = st.checkbox(
            "Code Quality Report",
            value=True,
            help="Generate additional code quality analysis"
        )
        
        st.markdown("---")

    st.markdown("## 🔍 Analyze Repository")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        # The text input's value is now controlled by st.session_state.repo_url
        st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/owner/repository",
            help="Enter a public GitHub repository URL",
            key="url_input_widget", 
            value=st.session_state.repo_url, 
            on_change=lambda: setattr(st.session_state, 'repo_url', st.session_state.url_input_widget)
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button(
            "🚀 Analyze",
            type="primary",
            use_container_width=True
        )
    
    with st.expander("💡 **Try Example Repositories**"):
        c1, c2, c3 = st.columns(3)
        if c1.button("Streamlit", use_container_width=True):
            st.session_state.repo_url = "https://github.com/streamlit/streamlit"
            st.rerun()
        if c2.button("FastAPI", use_container_width=True):
            st.session_state.repo_url = "https://github.com/tiangolo/fastapi"
            st.rerun()
        if c3.button("TensorFlow", use_container_width=True):
            st.session_state.repo_url = "https://github.com/tensorflow/tensorflow"
            st.rerun()

    if analyze_button:
        if not st.session_state.repo_url:
            st.warning("Please enter a repository URL.")
        elif not api_key:
            st.error("🚨 Please enter your  API Key in the sidebar.")
        else:
            # Clear previous analysis and chat history on new analysis
            st.session_state.analysis = None
            st.session_state.messages = [] 
            analysis, error = clone_and_analyze_repository(st.session_state.repo_url, max_commits)
            if error:
                st.error(error)
            else:
                st.session_state.analysis = analysis

    # Display analysis results if available
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        st.markdown("---")
        st.markdown(f"## 📊 Analysis for `{analysis['repo_name']}`")
        display_metrics(analysis)
        
        tabs = st.tabs(["🤖 AI Summary", "📈 Code & Commits", "🧐 Code Quality", "💬 Chat with Repo", "📖 README"])
        
        with tabs[0]: # AI Summary
            with st.spinner("🧠 Generating AI summary..."):
                summary, error = generate_ai_summary(analysis, api_key)
                if error:
                    st.error(error)
                else:
                    st.markdown(summary)
        
        with tabs[1]: # Code & Commits
            if enable_visualizations:
                st.plotly_chart(create_commit_timeline(analysis['commit_patterns']['commit_months']), use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_language_chart(analysis['languages']), use_container_width=True)
                    st.plotly_chart(create_activity_heatmap(analysis['commit_patterns']), use_container_width=True)
                with col2:
                    st.plotly_chart(create_contributor_chart(analysis['commit_patterns']['contributors']), use_container_width=True)
            
            display_code_explorer(analysis)
            display_commit_history(analysis['commit_data'])
            
        with tabs[2]: # Code Quality
            if enable_code_quality:
                with st.spinner("🔬 Assessing code quality..."):
                    report, error = generate_code_quality_report(analysis, api_key)
                    if error:
                        st.error(error)
                    else:
                        st.markdown(report)
            else:
                st.info("Enable 'Code Quality Report' in the sidebar to view this analysis.")

        with tabs[3]: # Chat with Repo
            st.markdown("### 💬 Chat with Repo")
            st.info("Ask questions about the repository's purpose, architecture, or how to get started.")

            # Display chat messages from history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("What is the main purpose of this project?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("AI is thinking..."):
                        response, error = generate_chat_response(
                            analysis, 
                            st.session_state.messages, 
                            prompt, 
                            api_key
                        )
                        if error:
                            st.error(error)
                        else:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

        with tabs[4]: # README
            if analysis['readme']:
                st.markdown(analysis['readme'], unsafe_allow_html=True)
            else:
                st.warning("No README file found in this repository.")

if __name__ == "__main__":
    main()
