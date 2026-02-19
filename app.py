"""
Semax Cognitive Testing Suite
==============================
A Streamlit-based cognitive testing application for tracking cognitive
performance during Semax supplementation.

SETUP INSTRUCTIONS:
1. Install Python 3.11+
2. Install dependencies: pip install -r requirements.txt
3. Get free Gemini API key: https://ai.google.dev/
4. Create .streamlit folder: mkdir .streamlit
5. Create secrets file: .streamlit/secrets.toml
6. Add to secrets.toml: GEMINI_API_KEY = "your_key_here"
7. Run: streamlit run app.py

FIRST RUN:
- Whisper model will auto-download (~150MB-1.5GB)
- Requires internet connection
- Takes 2-5 minutes

HARDWARE:
- GPU (CUDA): Faster transcription (2-5 seconds)
- CPU only: Slower but works (10-30 seconds)
- Automatically detects best available hardware

USAGE:
streamlit run app.py
"""

import streamlit as st
import whisper
import torch
from google import genai
import arxiv
import pandas as pd
import os
import json
import random
import string
import tempfile
import time
import re
from datetime import datetime
from streamlit_mic_recorder import mic_recorder
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables from .env file (if it exists)
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Semax Cognitive Testing",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .big-timer {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
        font-family: 'Courier New', monospace;
    }
    .digit-display {
        font-size: 72px;
        font-weight: bold;
        text-align: center;
        letter-spacing: 30px;
        color: #4B9AFF;
        font-family: 'Courier New', monospace;
        padding: 40px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        margin: 20px 0;
    }
    .countdown-text {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #FFD700;
    }
    .score-display {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
    }
    .success {
        background-color: #28a745;
        color: white;
    }
    .failure {
        background-color: #dc3545;
        color: white;
    }
    .attempt-counter {
        font-size: 24px;
        text-align: center;
        color: #888;
    }
    .phase-header {
        font-size: 28px;
        font-weight: bold;
        color: #4B9AFF;
        border-bottom: 2px solid #4B9AFF;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .progress-container {
        margin: 20px 0;
    }
    .stroop-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        padding: 30px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        margin: 20px 0;
    }
    .stroop-word {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 15px 10px;
        font-family: 'Arial', sans-serif;
        text-transform: uppercase;
    }
    .stroop-red { color: #FF4444; }
    .stroop-blue { color: #4488FF; }
    .stroop-green { color: #44DD44; }
    .stroop-yellow { color: #FFDD44; }
    .stroop-timer {
        font-size: 64px;
        font-weight: bold;
        text-align: center;
        color: #4B9AFF;
        font-family: 'Courier New', monospace;
        padding: 20px;
    }
    .stroop-timer-running {
        color: #44DD44;
    }
    .stroop-instructions {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# WHISPER MODEL LOADING (Cached) - BUG 1 FIX
# ═══════════════════════════════════════════════════════════════════════════════

# Initialize whisper status in session state BEFORE loading
if "whisper_status" not in st.session_state:
    st.session_state.whisper_status = "loading"

@st.cache_resource
def load_whisper_model():
    """Load Whisper Turbo model with automatic device detection."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use 'turbo' model for best speed/accuracy balance (~1.5GB download)
        model = whisper.load_model("turbo").to(device)
        return model, device, None
    except Exception as e:
        error_msg = str(e)
        # Add helpful context for common errors
        if "download" in error_msg.lower() or "connection" in error_msg.lower():
            error_msg = f"Failed to download Whisper model. Check internet connection. Original error: {error_msg}"
        return None, None, error_msg

# Load model at startup
whisper_model, whisper_device, whisper_error = load_whisper_model()

# Update status based on loading result
if whisper_error:
    st.session_state.whisper_status = "error"
    st.session_state.whisper_error_msg = whisper_error
elif whisper_model is not None:
    st.session_state.whisper_status = "active"
else:
    st.session_state.whisper_status = "error"

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_spoken_digits(transcript):
    """
    Convert Whisper output to digit list.
    Handles multiple formats:
    - "9 6 1 1 1" (space-separated)
    - "9-6-1-1-1" (hyphen-separated)
    - "9, 6, 1, 1, 1" (comma-separated)
    - "nine six one one one" (words)
    - "Nine six one one one" (capitalized words)
    """
    # Word to digit mapping
    word_to_digit = {
        "zero": 0, "oh": 0, "o": 0,
        "one": 1, "won": 1,
        "two": 2, "to": 2, "too": 2,
        "three": 3,
        "four": 4, "for": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8, "ate": 8,
        "nine": 9
    }
    
    # Lowercase the whole transcript
    text = transcript.lower()
    
    # Replace common separators with spaces
    for sep in ["-", "–", "—", ",", ".", ":", ";", "/", "\\"]:
        text = text.replace(sep, " ")
    
    # Split on whitespace
    tokens = text.split()
    
    digits = []
    for token in tokens:
        # Strip remaining punctuation
        token = token.strip(".,!?:;'\"")
        
        if not token:
            continue
        
        # Check if it's already a digit (or multi-digit like "123")
        if token.isdigit():
            for d in token:
                digits.append(int(d))
        # Check if it's a word in our mapping
        elif token in word_to_digit:
            digits.append(word_to_digit[token])
        # Ignore other tokens (filler words, etc.)
    
    return digits


def _clean_whisper_transcript(text):
    """
    WHISPER HALLUCINATION FIX: Strip trailing gibberish caused by silence.
    """
    if not text:
        return text
    
    # Common Whisper hallucination phrases that appear during silence
    hallucination_phrases = [
        "thank you for watching", "thanks for watching", "thank you for listening",
        "thanks for listening", "please subscribe", "like and subscribe",
        "see you next time", "goodbye", "bye bye", "bye-bye",
        "thank you", "you", "the end", "subtitles by",
        "i'll see you in the next video", "i'll see you in the next one",
    ]
    
    # Strip trailing hallucination phrases (case-insensitive)
    cleaned = text.strip()
    changed = True
    while changed:
        changed = False
        lower = cleaned.lower().rstrip('.!?, ')
        for phrase in hallucination_phrases:
            if lower.endswith(phrase):
                cleaned = cleaned[:len(cleaned) - (len(lower) - lower.rindex(phrase))].rstrip(' .,!?')
                changed = True
                break
    
    # Detect trailing repeated words (e.g., "word word word word")
    words = cleaned.split()
    if len(words) >= 4:
        last_word = words[-1].lower().strip('.,!?')
        repeat_count = 0
        for w in reversed(words):
            if w.lower().strip('.,!?') == last_word:
                repeat_count += 1
            else:
                break
        if repeat_count >= 3:
            cleaned = ' '.join(words[:len(words) - repeat_count + 1])
    
    return cleaned.strip()


def transcribe_audio(audio_bytes):
    """Transcribe audio using Whisper Turbo model."""
    # BUG 2 FIX: Check if audio bytes exist and are not empty
    if not audio_bytes or len(audio_bytes) == 0:
        return "ERROR: No audio data captured"
    
    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        # WHISPER HALLUCINATION FIX: condition_on_previous_text=False prevents
        # the model from using already-generated text to hallucinate more during silence
        result = whisper_model.transcribe(
            temp_path,
            condition_on_previous_text=False
        )
        
        # Cleanup
        os.unlink(temp_path)
        
        transcript = result["text"].strip()
        
        # BUG 2 FIX: Check if transcript is empty
        if not transcript:
            return "ERROR: No speech detected in audio"
        
        # WHISPER HALLUCINATION FIX: Clean trailing gibberish from silence
        transcript = _clean_whisper_transcript(transcript)
        
        return transcript
    except Exception as e:
        return f"ERROR: {str(e)}"


# Gemini model name - centralized for easy updates
GEMINI_MODEL = "gemini-3-flash-preview"

def call_gemini(prompt, api_key):
    """Call Gemini API using the new google-genai SDK."""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        return response.text
    except Exception as e:
        # IMPROVED ERROR: Show full error message from Google
        return f"ERROR: Gemini API failed - {str(e)}"


def parse_gemini_json(response):
    """
    BUG 7 FIX: Robust JSON parsing from Gemini response.
    Handles markdown code blocks and text mixed with JSON.
    """
    if response.startswith("ERROR:"):
        return None, response
    
    try:
        # First, try direct parsing
        return json.loads(response), None
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    response = response.strip()
    if "```" in response:
        # Extract content between ``` markers
        parts = response.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                try:
                    return json.loads(part), None
                except json.JSONDecodeError:
                    continue
    
    # Try finding JSON with regex
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, response)
    for match in matches:
        try:
            return json.loads(match), None
        except json.JSONDecodeError:
            continue
    
    return None, f"Could not parse JSON from response: {response[:200]}"


def fetch_random_arxiv_paper():
    """Fetch a random paper from arxiv across diverse scientific domains."""
    
    def _fetch_op(query, start_offset):
        # Disable arxiv library's internal retry/backoff (num_retries=0)
        client = arxiv.Client(num_retries=0, page_size=5)
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = list(client.results(search))
        if results:
            idx = min(start_offset, len(results) - 1)
            return results[idx]
        return None
    try:
        # Diverse search terms across ALL major arXiv categories
        search_terms = [
            # Physics
            "quantum mechanics", "astrophysics", "condensed matter", "particle physics",
            "thermodynamics", "optics", "plasma physics", "nuclear physics",
            # Mathematics
            "number theory", "algebraic geometry", "differential equations", "topology",
            "probability theory", "combinatorics", "mathematical analysis",
            # Quantitative Biology
            "computational biology", "genomics", "neuroscience", "population dynamics",
            "systems biology", "bioinformatics", "evolutionary biology",
            # Quantitative Finance
            "portfolio optimization", "risk management", "derivatives pricing",
            "algorithmic trading", "market microstructure",
            # Statistics
            "Bayesian statistics", "time series analysis", "causal inference",
            "hypothesis testing", "regression analysis",
            # Computer Science (keep some)
            "cryptography", "distributed systems", "programming languages",
            "computer graphics", "human computer interaction",
            # Economics
            "game theory", "behavioral economics", "econometrics",
            # Electrical Engineering
            "signal processing", "control systems", "communications",
        ]
        
        # Max retries with different queries
        max_retries = 3
        for attempt in range(max_retries):
            query = random.choice(search_terms)
            start_offset = random.randint(0, 4)
            
            # Use ThreadPoolExecutor to enforce a strict timeout
            # We explicitly do NOT use 'with' context manager to avoid waiting on shutdown
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_fetch_op, query, start_offset)
            
            try:
                paper = future.result(timeout=8)
                
                # Cleanup executor peacefully if successful
                executor.shutdown(wait=False)
                
                if paper:
                    return {
                        "id": paper.entry_id,
                        "title": paper.title,
                        "summary": paper.summary
                    }
            except concurrent.futures.TimeoutError:
                # Force shutdown of the stuck thread (fire and forget)
                executor.shutdown(wait=False)
                # Continue 'for' loop to try another query
                continue
            except Exception as e:
                executor.shutdown(wait=False)
                # Continue 'for' loop
                continue
        
        return {"error": "Could not fetch paper after multiple attempts (timeout)."}
        
    except Exception as e:
        return {"error": str(e)}


def save_to_csv(data):
    """Append test results to CSV file."""
    csv_path = "semax_cognitive_log.csv"
    df = pd.DataFrame([data])
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


def render_data_visualizations():
    """Render interactive Plotly graphs for longitudinal data analysis."""
    csv_path = "semax_cognitive_log.csv"
    
    if not os.path.exists(csv_path):
        return
    
    df = pd.read_csv(csv_path)
    
    # Need at least 2 days of data
    if len(df) < 2:
        st.info("📊 Complete at least 2 days to see trends")
        return
    
    st.markdown("---")
    st.subheader("📊 Data Visualizations")
    
    # Add day numbers (0-indexed)
    df['day'] = range(len(df))
    
    # Common layout settings for all graphs
    def add_cycle_shapes(fig, max_day):
        """Placeholder for cycle shading - returns figure unchanged."""
        # Cycle shading removed for generic use
        return fig
    
    common_layout = dict(
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # X-AXIS FIX: Use integer days only, not fractional
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    max_day = df['day'].max()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 1: Cognitive Performance (Raw Scores) - NEW
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🧠 Cognitive Performance (Raw Scores)")
    
    fig0 = go.Figure()
    
    # Add raw score traces
    fig0.add_trace(go.Scatter(x=df['day'], y=df['verbal_fluency_score'], name="Verbal Fluency (words)", mode='lines+markers', line=dict(color='#4B9AFF')))
    fig0.add_trace(go.Scatter(x=df['day'], y=df['digit_span_attempts'], name="Digit Span Total Attempts", mode='lines+markers', line=dict(color='#44DD44')))
    fig0.add_trace(go.Scatter(x=df['day'], y=df['synthesis_score'], name="Abstract Synthesis (/10)", mode='lines+markers', line=dict(color='#FFD93D')))
    fig0.add_trace(go.Scatter(x=df['day'], y=df['stroop_interference_effect'], name="Stroop Interference (sec)", mode='lines+markers', line=dict(color='#FF6B6B')))
    
    fig0 = add_cycle_shapes(fig0, max_day)
    fig0.update_layout(**common_layout, xaxis_title="Day", yaxis_title="Score")
    st.plotly_chart(fig0, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 2: Cognitive Performance (Normalized % Change from Baseline)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🧠 Cognitive Performance (% Change from Baseline)")
    
    fig1 = go.Figure()
    
    # Calculate baselines (Day 0 values)
    baseline_vf = df['verbal_fluency_score'].iloc[0] if df['verbal_fluency_score'].iloc[0] > 0 else 1
    baseline_ds = df['digit_span_time_seconds'].iloc[0] if df['digit_span_time_seconds'].iloc[0] > 0 else 1
    baseline_as = df['synthesis_score'].iloc[0] if df['synthesis_score'].iloc[0] > 0 else 1
    baseline_stroop = df['stroop_interference_effect'].iloc[0] if df['stroop_interference_effect'].iloc[0] > 0 else 1
    
    # Calculate % changes
    df['vf_pct'] = ((df['verbal_fluency_score'] - baseline_vf) / baseline_vf) * 100
    # Digit Span uses time - INVERT: lower time = better = positive change
    df['ds_pct'] = ((baseline_ds - df['digit_span_time_seconds']) / baseline_ds) * 100
    df['as_pct'] = ((df['synthesis_score'] - baseline_as) / baseline_as) * 100
    # Invert Stroop (lower is better)
    df['stroop_pct'] = -((df['stroop_interference_effect'] - baseline_stroop) / baseline_stroop) * 100
    
    # COLOR CONSISTENCY FIX: Use same colors as raw scores graph
    fig1.add_trace(go.Scatter(x=df['day'], y=df['vf_pct'], name="Verbal Fluency", mode='lines+markers', line=dict(color='#4B9AFF')))
    fig1.add_trace(go.Scatter(x=df['day'], y=df['ds_pct'], name="Digit Span (inverted)", mode='lines+markers', line=dict(color='#44DD44')))
    fig1.add_trace(go.Scatter(x=df['day'], y=df['as_pct'], name="Abstract Synthesis", mode='lines+markers', line=dict(color='#FFD93D')))
    fig1.add_trace(go.Scatter(x=df['day'], y=df['stroop_pct'], name="Stroop (inverted)", mode='lines+markers', line=dict(color='#FF6B6B')))
    
    fig1 = add_cycle_shapes(fig1, max_day)
    fig1.update_layout(**common_layout, xaxis_title="Day", yaxis_title="% Change from Baseline")
    fig1.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.3)
    st.plotly_chart(fig1, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 2: Stroop Performance (Raw Times)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🎨 Stroop Performance")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['day'], y=df['stroop_congruent_time'], name="Congruent (sec)", mode='lines+markers', line=dict(color='#44DD44')))
    fig2.add_trace(go.Scatter(x=df['day'], y=df['stroop_incongruent_time'], name="Incongruent (sec)", mode='lines+markers', line=dict(color='#FF6B6B')))
    fig2.add_trace(go.Scatter(x=df['day'], y=df['stroop_interference_effect'], name="Interference (sec)", mode='lines+markers', line=dict(color='#FFD93D')))
    
    fig2 = add_cycle_shapes(fig2, max_day)
    fig2.update_layout(**common_layout, xaxis_title="Day", yaxis_title="Time (seconds)")
    st.plotly_chart(fig2, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 3: Digit Span Performance
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🔢 Digit Span Performance")
    
    fig_ds = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Total Attempts on left axis
    fig_ds.add_trace(go.Scatter(x=df['day'], y=df['digit_span_attempts'], name="Total Attempts", mode='lines+markers', line=dict(color='#4B9AFF')), secondary_y=False)
    
    # Total Time on right axis
    fig_ds.add_trace(go.Scatter(x=df['day'], y=df['digit_span_time_seconds'], name="Total Time (sec)", mode='lines+markers', line=dict(color='#FF6B6B')), secondary_y=True)
    
    # Max Digits Completed (5, 6, or 7) on left axis
    fig_ds.add_trace(go.Scatter(x=df['day'], y=df['max_digit_span'], name="Max Digits Completed", mode='lines+markers', line=dict(color='#44DD44', dash='dash')), secondary_y=False)
    
    fig_ds = add_cycle_shapes(fig_ds, max_day)
    fig_ds.update_layout(**common_layout, xaxis_title="Day")
    fig_ds.update_yaxes(title_text="Count", secondary_y=False)
    fig_ds.update_yaxes(title_text="Time (seconds)", secondary_y=True)
    st.plotly_chart(fig_ds, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 4: Blood Pressure
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 🩸 Blood Pressure")
    
    # Parse BP strings (e.g., "120/80")
    def parse_bp(bp_str):
        try:
            parts = str(bp_str).split('/')
            return int(parts[0]), int(parts[1])
        except:
            return None, None
    
    df['morning_sys'], df['morning_dia'] = zip(*df['morning_bp'].apply(parse_bp))
    df['current_sys'], df['current_dia'] = zip(*df['current_bp'].apply(parse_bp))
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df['day'], y=df['morning_sys'], name="Morning Systolic", mode='lines+markers', line=dict(color='#FF6B6B')))
    fig3.add_trace(go.Scatter(x=df['day'], y=df['morning_dia'], name="Morning Diastolic", mode='lines+markers', line=dict(color='#FF6B6B', dash='dash')))
    fig3.add_trace(go.Scatter(x=df['day'], y=df['current_sys'], name="Current Systolic", mode='lines+markers', line=dict(color='#4B9AFF')))
    fig3.add_trace(go.Scatter(x=df['day'], y=df['current_dia'], name="Current Diastolic", mode='lines+markers', line=dict(color='#4B9AFF', dash='dash')))
    
    fig3 = add_cycle_shapes(fig3, max_day)
    fig3.update_layout(**common_layout, xaxis_title="Day", yaxis_title="mmHg")
    st.plotly_chart(fig3, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 4: Sleep Architecture
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### 😴 Sleep Architecture")
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df['day'], y=df['sleep_total_hours'], name="Total Sleep", mode='lines+markers', line=dict(color='#9C27B0')))
    fig4.add_trace(go.Scatter(x=df['day'], y=df['sleep_rem_hours'], name="REM Sleep", mode='lines+markers', line=dict(color='#00BCD4')))
    fig4.add_trace(go.Scatter(x=df['day'], y=df['sleep_deep_hours'], name="Deep Sleep", mode='lines+markers', line=dict(color='#3F51B5')))
    
    fig4 = add_cycle_shapes(fig4, max_day)
    fig4.update_layout(**common_layout, xaxis_title="Day", yaxis_title="Hours")
    st.plotly_chart(fig4, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH 5: Heart Rate & HRV (Dual Y-Axis)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("#### ❤️ Heart Rate & HRV")
    
    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Heart Rate on left axis
    fig5.add_trace(go.Scatter(x=df['day'], y=df['morning_hr'], name="Morning HR", mode='lines+markers', line=dict(color='#FF6B6B')), secondary_y=False)
    fig5.add_trace(go.Scatter(x=df['day'], y=df['current_hr'], name="Current HR", mode='lines+markers', line=dict(color='#FF9F43')), secondary_y=False)
    
    # HRV on right axis
    fig5.add_trace(go.Scatter(x=df['day'], y=df['morning_hrv_ms'], name="Morning HRV", mode='lines+markers', line=dict(color='#26C6DA')), secondary_y=True)
    
    fig5 = add_cycle_shapes(fig5, max_day)
    fig5.update_layout(**common_layout, xaxis_title="Day")
    fig5.update_yaxes(title_text="Heart Rate (BPM)", secondary_y=False)
    fig5.update_yaxes(title_text="HRV (ms)", secondary_y=True)
    st.plotly_chart(fig5, use_container_width=True)


def generate_digit_sequence(length):
    """Generate a random sequence of digits."""
    return [random.randint(0, 9) for _ in range(length)]


def format_digits_display(digits):
    """Format digits for large display."""
    return "  ".join(str(d) for d in digits)


# Stroop Test color configurations
STROOP_COLORS = ["red", "blue", "green", "yellow"]
STROOP_WORDS = ["RED", "BLUE", "GREEN", "YELLOW"]

def generate_stroop_grid(congruent=True):
    """
    Generate a 5x5 grid of Stroop words with balanced color distribution.
    
    Args:
        congruent: If True, word matches ink color. If False, word ≠ ink color.
    
    Returns:
        List of 25 tuples: (word, ink_color) in ROW-MAJOR order for CSS grid rendering.
        The grid is rendered left-to-right, top-to-bottom, but the USER reads it
        COLUMN by COLUMN (top-to-bottom per column, then next column).
    
    Constraints:
        - Each color appears 6-7 times (balanced distribution)
        - Max 2 consecutive same INK COLORS when reading column-by-column
    
    Grid layout (indices in row-major order for CSS):
        Col 0   Col 1   Col 2   Col 3   Col 4
        [0]     [1]     [2]     [3]     [4]      <- Row 0
        [5]     [6]     [7]     [8]     [9]      <- Row 1
        [10]    [11]    [12]    [13]    [14]     <- Row 2
        [15]    [16]    [17]    [18]    [19]     <- Row 3
        [20]    [21]    [22]    [23]    [24]     <- Row 4
    
    Reading order (column-by-column, top-to-bottom):
        0, 5, 10, 15, 20, 1, 6, 11, 16, 21, 2, 7, 12, 17, 22, 3, 8, 13, 18, 23, 4, 9, 14, 19, 24
    """
    GRID_SIZE = 5
    TOTAL_CELLS = GRID_SIZE * GRID_SIZE  # 25
    
    # Create balanced distribution: 6+6+7+6 = 25 colors
    color_pool = (STROOP_COLORS * 6) + [STROOP_COLORS[2]]  # 6 each + 1 extra green
    
    def get_column_reading_order():
        """Get indices in column-by-column reading order (top-to-bottom per column)."""
        order = []
        for col in range(GRID_SIZE):
            for row in range(GRID_SIZE):
                order.append(row * GRID_SIZE + col)
        return order
    
    def row_major_to_column_order(row_major_list):
        """Convert row-major list to column reading order."""
        order = get_column_reading_order()
        return [row_major_list[i] for i in order]
    
    def column_order_to_row_major(column_order_list):
        """Convert column reading order back to row-major for CSS grid."""
        order = get_column_reading_order()
        result = [None] * TOTAL_CELLS
        for i, idx in enumerate(order):
            result[idx] = column_order_list[i]
        return result
    
    def has_too_many_consecutive(colors, max_consecutive=2):
        """Check if any color appears more than max_consecutive times in a row."""
        if len(colors) < 2:
            return False
        count = 1
        for i in range(1, len(colors)):
            if colors[i] == colors[i-1]:
                count += 1
                if count > max_consecutive:
                    return True
            else:
                count = 1
        return False
    
    def shuffle_for_column_reading(colors, max_consecutive=2):
        """
        Shuffle colors ensuring no more than max_consecutive same colors 
        when reading in COLUMN order (top-to-bottom per column).
        Returns colors in ROW-MAJOR order for CSS grid rendering.
        """
        
        # Method 1: Try random shuffles and check column reading order
        for _ in range(1000):
            random.shuffle(colors)
            # Check in column reading order
            column_order = row_major_to_column_order(colors)
            if not has_too_many_consecutive(column_order, max_consecutive):
                return colors
        
        # Method 2: Build directly in column reading order, then convert
        remaining = list(colors)
        random.shuffle(remaining)
        column_result = []
        
        while remaining:
            # Find colors that won't create too many consecutive
            valid_colors = []
            for i, color in enumerate(remaining):
                # Count how many of this color are at the end of column_result
                consecutive_count = 0
                for j in range(len(column_result) - 1, -1, -1):
                    if column_result[j] == color:
                        consecutive_count += 1
                    else:
                        break
                
                if consecutive_count < max_consecutive:
                    valid_colors.append(i)
            
            if valid_colors:
                idx = random.choice(valid_colors)
                column_result.append(remaining.pop(idx))
            else:
                # Try insertion at earlier positions
                color_to_insert = remaining[0]
                inserted = False
                
                for insert_pos in range(len(column_result), -1, -1):
                    test = column_result[:insert_pos] + [color_to_insert] + column_result[insert_pos:]
                    if not has_too_many_consecutive(test, max_consecutive):
                        column_result = test
                        remaining.pop(0)
                        inserted = True
                        break
                
                if not inserted:
                    column_result.append(remaining.pop(0))
        
        # Final repair pass
        for _ in range(200):
            if not has_too_many_consecutive(column_result, max_consecutive):
                break
            
            # Find and fix problematic consecutive runs
            for i in range(len(column_result) - max_consecutive):
                run_length = 1
                while i + run_length < len(column_result) and column_result[i + run_length] == column_result[i]:
                    run_length += 1
                
                if run_length > max_consecutive:
                    problem_color = column_result[i]
                    # Find a swap target with different color
                    for swap_target in range(len(column_result)):
                        if column_result[swap_target] != problem_color:
                            swap_from = i + max_consecutive
                            if swap_from < len(column_result):
                                test = list(column_result)
                                test[swap_from], test[swap_target] = test[swap_target], test[swap_from]
                                if not has_too_many_consecutive(test, max_consecutive):
                                    column_result = test
                                    break
                    break
        
        # Convert back to row-major order for CSS grid
        return column_order_to_row_major(column_result)
    
    shuffled_colors = shuffle_for_column_reading(list(color_pool), max_consecutive=2)
    
    # Build grid with words
    grid = []
    for ink_color in shuffled_colors:
        if congruent:
            # Word matches ink color
            word_idx = STROOP_COLORS.index(ink_color)
            word = STROOP_WORDS[word_idx]
        else:
            # Word does NOT match ink color - pick a different word
            available_words = [(w, i) for i, w in enumerate(STROOP_WORDS) if STROOP_COLORS[i] != ink_color]
            word, _ = random.choice(available_words)
        
        grid.append((word, ink_color))
    
    return grid


def render_stroop_grid(grid):
    """Render Stroop grid as HTML."""
    html = '<div class="stroop-grid">'
    for word, color in grid:
        html += f'<span class="stroop-word stroop-{color}">{word}</span>'
    html += '</div>'
    return html


def format_stroop_time(seconds, timeout=False):
    """Format time for Stroop display."""
    if timeout:
        return f"{seconds:.1f}s (TIMEOUT)"
    return f"{seconds:.1f}s"


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Phase tracking
        "current_phase": 1,
        
        # Phase 1: Morning Vitals (Pre-Semax)
        "morning_bp": "120/80",
        "morning_hr": 70,
        "morning_spo2": 98,
        "sleep_total": 7.0,
        "sleep_total_hours": 7,
        "sleep_total_minutes": 0,
        "sleep_rem": 1.5,
        "sleep_rem_hours": 1,
        "sleep_rem_minutes": 30,
        "sleep_deep": 1.5,
        "sleep_deep_hours": 1,
        "sleep_deep_minutes": 30,
        "morning_hrv": 50,
        "morning_timestamp": "",
        
        # Phase 1: Current Vitals (Pre-Test)
        "current_bp": "120/80",
        "current_hr": 70,
        "current_spo2": 98,
        "hours_since_semax": 1.0,
        "total_dose_mcg": 400,
        "current_timestamp": "",
        
        # Phase 1: Subjective Notes
        "subjective_notes": "",
        "subjective_audio_processed": False,
        
        # Phase 2: Verbal Fluency
        "vf_letter": None,
        "vf_started": False,
        "vf_timer_end": None,
        "vf_transcript": "",
        "vf_score": None,
        "vf_invalid_words": [],
        "vf_recording_complete": False,
        "vf_error": None,  # BUG 2 FIX: Track errors
        "vf_grading_error": None,  # Track Gemini grading errors separately
        "vf_audio_bytes": None,  # Store audio bytes during auto-recording
        
        # Phase 3: Digit Span
        "ds_level": 1,  # 1=5 digits, 2=6 digits, 3=7 digits
        "ds_current_sequence": None,
        "ds_tested_sequence": None,  # BUG 3 FIX: Store the sequence being tested
        "ds_showing_digits": False,
        "ds_digits_hidden": False,
        "ds_attempts": 0,
        "ds_total_attempts": 0,
        "ds_max_achieved": 0,
        "ds_test_complete": False,
        "ds_last_result": None,
        "ds_user_answer": [],
        "ds_transcript": "",
        "ds_error": None,  # BUG 8 FIX: Track errors
        "ds_answer_start_time": None,  # Track time when answer recording starts
        "ds_recording_start_time": None,  # TIMER FIX: Store when recording starts
        "ds_recording_end_time": None,  # TIMER FIX: Store when recording ends  
        "ds_last_attempt_time": 0.0,  # TIMER FIX: Time for last attempt
        "ds_total_time": 0.0,  # Total time across all attempts (sum)
        "ds_level1_time": 0.0,  # PER-LEVEL TIMING: Store successful time for level 1 (5 digits)
        "ds_level2_time": 0.0,  # PER-LEVEL TIMING: Store successful time for level 2 (6 digits)
        "ds_level3_time": 0.0,  # PER-LEVEL TIMING: Store successful time for level 3 (7 digits)
        
        # Phase 4: Abstract Synthesis
        "as_paper": None,
        "as_reading_started": False,
        "as_reading_end": None,
        "as_abstract_hidden": False,
        "as_transcript": "",
        "as_score": None,
        "as_feedback": "",
        "as_recording_complete": False,
        "as_explanation_started": False,  # BUG 6 FIX: Track explanation recording
        "as_explanation_end": None,  # BUG 6 FIX: Countdown for explanation
        "as_error": None,  # BUG 8 FIX: Track errors
        "as_grading_error": None,  # BUG 7 FIX: Track grading errors
        
        # API Key
        "gemini_api_key": "",
        
        # Recording states
        "recording_phase": None,
        
        # Phase 5: Stroop Test
        "stroop_stage": "instructions",  # instructions, congruent, congruent_result, incongruent, incongruent_result, complete
        "stroop_congruent_grid": None,
        "stroop_incongruent_grid": None,
        "stroop_congruent_time": None,
        "stroop_incongruent_time": None,
        "stroop_timer_start": None,
        "stroop_timer_running": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - API KEY MANAGEMENT & STATUS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Try to load API key from multiple sources (in priority order):
    # 1. Streamlit secrets (for Streamlit Cloud deployment)
    # 2. Environment variable / .env file (for local development)
    # 3. Manual input (fallback)
    
    api_key = None
    
    # Method 1: Streamlit secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.gemini_api_key = api_key
            st.success("✅ API Key loaded from secrets")
    except Exception:
        pass  # secrets.toml doesn't exist
    
    # Method 2: Environment variable (from .env or system)
    if not api_key:
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            api_key = env_key
            st.session_state.gemini_api_key = api_key
            st.success("✅ API Key loaded from .env")
    
    # Method 3: Manual input
    if not api_key:
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Enter your Google Gemini API key. Get one free at https://ai.google.dev/"
        )
        st.session_state.gemini_api_key = api_key
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API key required. Get free key at: https://ai.google.dev/")
    
    st.divider()
    
    # Progress indicator
    st.subheader("📊 Progress")
    phases = ["Vitals", "Stroop Test", "Verbal Fluency", "Digit Span", "Abstract Synthesis", "Results"]
    for i, phase in enumerate(phases, 1):
        if i < st.session_state.current_phase:
            st.write(f"✅ Phase {i}: {phase}")
        elif i == st.session_state.current_phase:
            st.write(f"▶️ Phase {i}: {phase}")
        else:
            st.write(f"⬜ Phase {i}: {phase}")
    
    # SKIP TEST BUTTON - visible during test phases (2-5)
    current = st.session_state.current_phase
    if current in [2, 3, 4, 5]:
        phase_names = {2: "Stroop Test", 3: "Verbal Fluency", 4: "Digit Span", 5: "Abstract Synthesis"}
        st.warning(f"⚠️ Currently in: **{phase_names[current]}**")
        if st.button("⏭️ Skip Test", key="skip_test_btn", type="secondary", use_container_width=True):
            
            if current == 2:  # Skip Stroop
                st.session_state.stroop_congruent_time = None
                st.session_state.stroop_incongruent_time = None
                st.session_state.stroop_congruent_start = None
                st.session_state.stroop_incongruent_start = None
                # Initialize Phase 3 VF letter (normally set during Stroop→VF transition)
                st.session_state.vf_letter = random.choice(string.ascii_uppercase)
            
            elif current == 3:  # Skip Verbal Fluency
                st.session_state.vf_score = 0
                st.session_state.vf_transcript = ""
                st.session_state.vf_feedback = ""
                st.session_state.vf_recording_complete = False
                st.session_state.vf_grading_complete = False
            
            elif current == 4:  # Skip Digit Span
                st.session_state.ds_total_attempts = 0
                st.session_state.ds_max_achieved = 0
                st.session_state.ds_total_time = 0.0
                st.session_state.ds_test_complete = True
                st.session_state.ds_level1_time = 0.0
                st.session_state.ds_level2_time = 0.0
                st.session_state.ds_level3_time = 0.0
                st.session_state.ds_showing_digits = False
                st.session_state.ds_digits_hidden = False
                st.session_state.ds_last_result = None
                st.session_state.ds_transcript = ""
                st.session_state.ds_user_answer = []
            
            elif current == 5:  # Skip Abstract Synthesis
                st.session_state.as_score = 0
                st.session_state.as_transcript = ""
                st.session_state.as_feedback = ""
                st.session_state.as_paper = None
                st.session_state.as_recording_complete = False
                st.session_state.as_grading_complete = False
            
            # Advance to next phase
            st.session_state.current_phase = current + 1
            st.rerun()
        
        st.caption("Skipped tests will show blank/zero values in results and CSV.")
    
    st.divider()
    
    # Device info - BUG 1 FIX: Show Whisper status here instead of banner
    st.subheader("🖥️ System Info")
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.write(f"Device: {device}")
    if torch.cuda.is_available():
        st.write(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # BUG 1 FIX: Whisper status indicator
    if st.session_state.whisper_status == "active":
        device_name = "CUDA" if whisper_device == "cuda" else "CPU"
        st.success(f"Whisper Status: ✅ Active ({device_name})")
    elif st.session_state.whisper_status == "loading":
        st.info("Whisper Status: ⏳ Loading...")
    else:
        st.error("Whisper Status: ❌ Error")
        if hasattr(st.session_state, 'whisper_error_msg'):
            st.caption(st.session_state.whisper_error_msg)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🧠 Semax Cognitive Testing Suite")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: VITALS INTAKE
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.current_phase == 1:
    st.markdown('<p class="phase-header">Phase 1: Vitals Intake</p>', unsafe_allow_html=True)
    
    # CSS to constrain vitals container and prevent stretching in wide mode
    st.markdown("""
    <style>
        /* Constrain main content area */
        .block-container {
            max-width: 1200px !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* Force max-width on number inputs in vitals */
        .stNumberInput input {
            max-width: 120px !important;
        }
        
        /* Force max-width on text inputs */
        .stTextInput input {
            max-width: 180px !important;
        }
        
        /* Prevent column stretching */
        [data-testid="column"] {
            max-width: 300px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MORNING VITALS (Pre-Semax)
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("🌅 Morning Vitals (Pre-Semax)")
    
    # Blood Pressure, Heart Rate, SpO2 - 3 columns (only place we use columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input(
            "🩸 Blood Pressure", value=st.session_state.morning_bp,
            placeholder="120/80", key="morning_bp_input"
        )
        st.session_state.morning_bp = st.session_state.morning_bp_input
    with col2:
        st.session_state.morning_hr = st.number_input(
            "❤️ Heart Rate (BPM)", min_value=40, max_value=200,
            value=st.session_state.morning_hr, key="morning_hr_input"
        )
    with col3:
        st.session_state.morning_spo2 = st.number_input(
            "💨 SpO2 (%)", min_value=70, max_value=100,
            value=st.session_state.morning_spo2, key="morning_spo2_input"
        )
    
    # Sleep Data - simplified inline layout
    st.markdown("#### 😴 Sleep Data")
    
    # Total Sleep - label on top, inputs below
    st.markdown("**Total Sleep:**")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sleep_total_h = st.number_input("Hours", min_value=0, max_value=24, 
            value=st.session_state.sleep_total_hours, key="sleep_total_h")
    with c2:
        sleep_total_m = st.number_input("Minutes", min_value=0, max_value=59,
            value=st.session_state.sleep_total_minutes, key="sleep_total_m")
    with c3:
        st.session_state.sleep_total_hours = sleep_total_h
        st.session_state.sleep_total_minutes = sleep_total_m
        st.session_state.sleep_total = round(sleep_total_h + (sleep_total_m / 60), 2)
        st.markdown(f'<p style="margin-top: 28px;">= <strong>{st.session_state.sleep_total} hours</strong></p>', unsafe_allow_html=True)
    
    # REM Sleep - label on top, inputs below
    st.markdown("**💭 REM Sleep:**")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sleep_rem_h = st.number_input("Hours", min_value=0, max_value=12,
            value=st.session_state.sleep_rem_hours, key="sleep_rem_h")
    with c2:
        sleep_rem_m = st.number_input("Minutes", min_value=0, max_value=59,
            value=st.session_state.sleep_rem_minutes, key="sleep_rem_m")
    with c3:
        st.session_state.sleep_rem_hours = sleep_rem_h
        st.session_state.sleep_rem_minutes = sleep_rem_m
        st.session_state.sleep_rem = round(sleep_rem_h + (sleep_rem_m / 60), 2)
        st.markdown(f'<p style="margin-top: 28px;">= <strong>{st.session_state.sleep_rem} hours</strong></p>', unsafe_allow_html=True)
    
    # Deep Sleep - label on top, inputs below
    st.markdown("**🌙 Deep Sleep:**")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        sleep_deep_h = st.number_input("Hours", min_value=0, max_value=12,
            value=st.session_state.sleep_deep_hours, key="sleep_deep_h")
    with c2:
        sleep_deep_m = st.number_input("Minutes", min_value=0, max_value=59,
            value=st.session_state.sleep_deep_minutes, key="sleep_deep_m")
    with c3:
        st.session_state.sleep_deep_hours = sleep_deep_h
        st.session_state.sleep_deep_minutes = sleep_deep_m
        st.session_state.sleep_deep = round(sleep_deep_h + (sleep_deep_m / 60), 2)
        st.markdown(f'<p style="margin-top: 28px;">= <strong>{st.session_state.sleep_deep} hours</strong></p>', unsafe_allow_html=True)
    
    # HRV - single input (not in columns)
    st.session_state.morning_hrv = st.number_input(
        "💓 Morning HRV (ms)", min_value=0, max_value=300,
        value=st.session_state.morning_hrv, key="morning_hrv_input"
    )
    
    # Morning Time - 12-hour format
    default_morning_time = datetime.now().strftime("%I:%M%p").lower().lstrip("0")
    st.session_state.morning_timestamp = st.text_input(
        "⏰ Morning Time",
        value=st.session_state.morning_timestamp if st.session_state.morning_timestamp else default_morning_time,
        key="morning_timestamp_input"
    )
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CURRENT VITALS (Pre-Test)
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("🧪 Current Vitals (Pre-Test)")
    
    # Blood Pressure, Heart Rate, SpO2 - 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text_input(
            "🩸 Blood Pressure", value=st.session_state.current_bp,
            placeholder="120/80", key="current_bp_input"
        )
        st.session_state.current_bp = st.session_state.current_bp_input
    with col2:
        st.session_state.current_hr = st.number_input(
            "❤️ Heart Rate (BPM)", min_value=40, max_value=200,
            value=st.session_state.current_hr, key="current_hr_input"
        )
    with col3:
        st.session_state.current_spo2 = st.number_input(
            "💨 SpO2 (%)", min_value=70, max_value=100,
            value=st.session_state.current_spo2, key="current_spo2_input"
        )
    
    # Hours Since Semax + Total Dose - side by side
    dose_col1, dose_col2 = st.columns(2)
    with dose_col1:
        st.session_state.hours_since_semax = st.number_input(
            "💊 Hours Since Semax Dose", min_value=0.0, max_value=48.0,
            value=st.session_state.hours_since_semax, step=0.5, key="hours_since_semax_input"
        )
    with dose_col2:
        st.session_state.total_dose_mcg = st.number_input(
            "💉 Total Dose Amount (mcg)", min_value=0, max_value=5000,
            value=st.session_state.total_dose_mcg, step=100, key="total_dose_mcg_input"
        )
    
    # Current Time - 12-hour format, auto-captured
    current_time_12hr = datetime.now().strftime("%I:%M%p").lower().lstrip("0")
    st.session_state.current_timestamp = st.text_input(
        "⏰ Current Time (auto-captured)",
        value=current_time_12hr, key="current_timestamp_input"
    )
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUBJECTIVE NOTES
    # ═══════════════════════════════════════════════════════════════════════════
    st.subheader("🎤 How do you feel?")
    st.write("Record your subjective notes (up to 120 seconds):")
    
    # Audio recorder for subjective notes
    audio = mic_recorder(
        start_prompt="🔴 RECORD SUBJECTIVE NOTES",
        stop_prompt="⏹️ STOP RECORDING",
        just_once=False,
        key="subjective_recorder"
    )
    
    if audio and not st.session_state.subjective_audio_processed:
        with st.spinner("🔄 Transcribing audio... (8-15 seconds)"):
            transcript = transcribe_audio(audio["bytes"])
            if not transcript.startswith("ERROR"):
                st.session_state.subjective_notes = transcript
                st.session_state.subjective_audio_processed = True
            else:
                st.error(f"Transcription error: {transcript}")
    
    # Editable transcript
    st.session_state.subjective_notes = st.text_area(
        "Transcript (editable):",
        value=st.session_state.subjective_notes,
        height=100
    )
    
    if st.button("Re-record", key="rerecord_subjective"):
        st.session_state.subjective_audio_processed = False
        st.rerun()
    
    st.divider()
    
    if st.button("▶️ START COGNITIVE TESTS", type="primary", use_container_width=True):
        st.session_state.current_phase = 2
        # Initialize Stroop grids for Phase 2
        st.session_state.stroop_congruent_grid = generate_stroop_grid(congruent=True)
        st.session_state.stroop_incongruent_grid = generate_stroop_grid(congruent=False)
        st.session_state.stroop_stage = "instructions"
        st.session_state.stroop_timer_start = None
        st.session_state.stroop_timer_running = False
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: VERBAL FLUENCY TEST
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_phase == 3:
    st.markdown('<p class="phase-header">Phase 3: Verbal Fluency Test</p>', unsafe_allow_html=True)
    
    letter = st.session_state.vf_letter
    
    # Show error and retry button if there was an error
    if st.session_state.vf_error:
        st.error(f"❌ Error: {st.session_state.vf_error}")
        if st.button("🔄 RETRY TEST", type="primary", use_container_width=True, key="vf_retry"):
            # Reset Phase 2 state
            st.session_state.vf_started = False
            st.session_state.vf_timer_end = None
            st.session_state.vf_transcript = ""
            st.session_state.vf_score = None
            st.session_state.vf_invalid_words = []
            st.session_state.vf_recording_complete = False
            st.session_state.vf_error = None
            st.session_state.vf_letter = random.choice(string.ascii_uppercase)
            st.rerun()
    
    elif not st.session_state.vf_recording_complete:
        # Show the letter prominently
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    border-radius: 15px; margin: 20px 0;">
            <span style="font-size: 20px; color: #888;">Name as many words starting with</span><br>
            <span style="font-size: 96px; font-weight: bold; color: #4B9AFF; font-family: 'Courier New', monospace;">{letter}</span><br>
            <span style="font-size: 20px; color: #888;">as you can in 60 seconds</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize recording_started flag if not exists
        if "vf_recording_started" not in st.session_state:
            st.session_state.vf_recording_started = False
        
        # STAGE 3: Timer is running - show countdown
        if st.session_state.vf_started and st.session_state.vf_timer_end:
            remaining = max(0, st.session_state.vf_timer_end - time.time())
            
            # LARGE COUNTDOWN TIMER (48px, yellow)
            if remaining > 0:
                st.markdown(f'<p class="countdown-text" style="font-size: 48px; color: #FFD700;">⏱️ {int(remaining)} seconds</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="countdown-text" style="font-size: 48px; color: #FF4444;">⏱️ Time\'s up! Click STOP RECORDING.</p>', unsafe_allow_html=True)
            
            st.markdown("### 🔴 RECORDING IN PROGRESS - Speak now!")
        
        # STAGE 2: Recording started, waiting for timer start
        elif st.session_state.vf_recording_started:
            st.markdown('<p class="countdown-text" style="font-size: 48px; color: #00FF00;">🎙️ Recording... Click START TIMER when ready!</p>', unsafe_allow_html=True)
            
            if st.button("🎯 START TIMER (60 seconds)", type="primary", use_container_width=True, key="vf_start_timer"):
                st.session_state.vf_started = True
                st.session_state.vf_timer_end = time.time() + 60
                st.rerun()
        
        # STAGE 1: Not recording yet - show instructions
        else:
            st.markdown("### 📋 Instructions:")
            st.markdown("1. Click **START RECORDING** below")
            st.markdown("2. Then click **START TIMER** when you're ready to begin")
            st.markdown("3. Name words for 60 seconds, then click **STOP RECORDING**")
        
        st.divider()
        
        # VF UI FIX: Initialize grading flag
        if "vf_grading_in_progress" not in st.session_state:
            st.session_state.vf_grading_in_progress = False
        
        # VF UI FIX: Only show recorder when NOT grading
        if not st.session_state.vf_grading_in_progress:
            # MIC RECORDER - visible only during active recording phase
            audio = mic_recorder(
                start_prompt="🔴 START RECORDING",
                stop_prompt="⏹️ STOP RECORDING",
                just_once=True,
                key="vf_recorder"
            )
            
            # Button to confirm recording has started (since we can't detect mic_recorder state)
            if not st.session_state.vf_recording_started:
                if st.button("✅ I clicked START RECORDING", use_container_width=True, key="vf_confirm_recording"):
                    st.session_state.vf_recording_started = True
                    st.rerun()
            
            # Process audio when user stops recording
            if audio and audio.get("bytes") and len(audio["bytes"]) > 0:
                # VF UI FIX: Store audio bytes in session state BEFORE rerun
                st.session_state.vf_audio_bytes = audio["bytes"]
                st.session_state.vf_grading_in_progress = True
                st.rerun()
        else:
            # VF UI FIX: Show grading spinner instead of recorder
            with st.spinner("🔄 Transcribing audio... (8-15 seconds)"):
                # Use stored audio bytes from session state
                if st.session_state.vf_audio_bytes:
                    transcript = transcribe_audio(st.session_state.vf_audio_bytes)
                    if not transcript.startswith("ERROR"):
                        st.session_state.vf_transcript = transcript
                    else:
                        st.session_state.vf_error = transcript.replace("ERROR: ", "")
                    st.session_state.vf_recording_complete = True
                    st.session_state.vf_grading_in_progress = False
                    st.session_state.vf_audio_bytes = None  # Clear stored audio
            st.rerun()
        
        # Auto-refresh countdown every second (only when timer is running)
        if st.session_state.vf_started and st.session_state.vf_timer_end:
            remaining = max(0, st.session_state.vf_timer_end - time.time())
            if remaining > 0:
                time.sleep(1)
                st.rerun()
    
    else:
        # Recording complete - show results and grade
        
        # Check for empty transcript
        if not st.session_state.vf_transcript and not st.session_state.vf_error:
            st.session_state.vf_error = "No speech was detected. Please try again and speak clearly."
            st.rerun()
        
        # Show transcript
        st.subheader("📝 Your Transcript:")
        st.info(st.session_state.vf_transcript)
        
        # Show grading error with retry button (mirrors Abstract Synthesis pattern)
        if st.session_state.vf_grading_error:
            st.error(f"❌ Grading Error: {st.session_state.vf_grading_error}")
            if st.button("🔄 RETRY GRADING", type="primary", use_container_width=True, key="vf_retry_grading"):
                st.session_state.vf_score = None
                st.session_state.vf_invalid_words = []
                st.session_state.vf_grading_error = None
                st.rerun()
        
        # Grade with Gemini
        elif st.session_state.vf_score is None and st.session_state.gemini_api_key:
            with st.spinner("🤖 Grading with Gemini..."):
                prompt = f"""You are grading a verbal fluency test.
The subject had 60 seconds to name words starting with the letter {letter}.
Transcript: {st.session_state.vf_transcript}

Count ONLY unique, valid English words starting with {letter}. 
Ignore filler words (um, uh, like, okay, and, the, etc).

PHONETIC LENIENCY (Whisper transcription quirk):
- If a word appears as a proper noun but has a valid common word homophone, count it as valid.
- Examples: 'Matt' → count as 'mat' (doormat), 'Mary' → count as 'marry' (verb), 'May' → 'may' (modal verb)
- Do NOT accept pure proper nouns with no common word equivalent (e.g., 'Max' is invalid - not same as 'maximum')

Return ONLY valid JSON (no markdown):
{{"valid_count": N, "invalid_words": ["list", "of", "invalid"]}}"""
                
                response = call_gemini(prompt, st.session_state.gemini_api_key)
                
                result, error = parse_gemini_json(response)
                if result:
                    st.session_state.vf_score = result.get("valid_count", 0)
                    st.session_state.vf_invalid_words = result.get("invalid_words", [])
                    st.session_state.vf_grading_success = True
                else:
                    st.session_state.vf_grading_error = error or "Failed to parse grading response"
            st.rerun()
        
        if st.session_state.vf_score is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="score-display success">Score: {st.session_state.vf_score} valid words</div>', 
                           unsafe_allow_html=True)
            with col2:
                if st.session_state.vf_invalid_words:
                    st.warning(f"Invalid words: {', '.join(st.session_state.vf_invalid_words)}")
            
            if st.session_state.get("vf_grading_success"):
                st.success(f"✅ Graded by {GEMINI_MODEL}")
        
        # NEXT TEST button
        if st.button("▶️ NEXT TEST", type="primary", use_container_width=True):
            st.session_state.current_phase = 4
            st.session_state.ds_current_sequence = generate_digit_sequence(5)
            st.session_state.ds_tested_sequence = st.session_state.ds_current_sequence.copy()
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: REVERSE DIGIT SPAN TEST
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_phase == 4:
    st.markdown('<p class="phase-header">Phase 4: Reverse Digit Span Test</p>', unsafe_allow_html=True)
    
    level = st.session_state.ds_level
    num_digits = 4 + level  # Level 1=5, Level 2=6, Level 3=7
    
    # BUG 8 FIX: Show error and retry button if there was an error
    if st.session_state.ds_error:
        st.error(f"❌ Error: {st.session_state.ds_error}")
        if st.button("🔄 RETRY TEST", type="primary", use_container_width=True, key="ds_retry"):
            # Reset Phase 3 state
            st.session_state.ds_level = 1
            st.session_state.ds_current_sequence = generate_digit_sequence(5)
            st.session_state.ds_tested_sequence = st.session_state.ds_current_sequence.copy()
            st.session_state.ds_showing_digits = False
            st.session_state.ds_digits_hidden = False
            st.session_state.ds_attempts = 0
            st.session_state.ds_total_attempts = 0
            st.session_state.ds_max_achieved = 0
            st.session_state.ds_test_complete = False
            st.session_state.ds_last_result = None
            st.session_state.ds_user_answer = []
            st.session_state.ds_transcript = ""
            st.session_state.ds_error = None
            st.rerun()
    
    elif not st.session_state.ds_test_complete:
        # CRITICAL: Double-check phase to prevent UI bleeding into Phase 5
        if st.session_state.current_phase != 4:
            pass  # Not in Phase 4, skip all rendering
        else:
            st.markdown(f'<p class="attempt-counter">Level {level}/3 • {num_digits} digits • Attempt {st.session_state.ds_attempts + 1}/5</p>', 
                       unsafe_allow_html=True)
            
            # Generate new sequence if needed
            if st.session_state.ds_current_sequence is None:
                st.session_state.ds_current_sequence = generate_digit_sequence(num_digits)
                st.session_state.ds_tested_sequence = st.session_state.ds_current_sequence.copy()  # BUG 3 FIX
            
            sequence = st.session_state.ds_current_sequence
            correct_answer = list(reversed(sequence))
            
            # Show digits phase
            if not st.session_state.ds_digits_hidden:
                if not st.session_state.ds_showing_digits:
                    st.write("Memorize these digits, then recite them **BACKWARDS**:")
                    if st.button("👀 SHOW DIGITS", type="primary"):
                        st.session_state.ds_showing_digits = True
                        st.session_state.ds_show_start = time.time()
                        # BUG 3 FIX: Store the sequence being tested BEFORE showing
                        st.session_state.ds_tested_sequence = st.session_state.ds_current_sequence.copy()
                        st.rerun()
                else:
                    elapsed = time.time() - st.session_state.ds_show_start
                    remaining = max(0, 5 - elapsed)
                    
                    if remaining > 0:
                        st.markdown(f'<p class="countdown-text">{int(remaining + 1)}...</p>', unsafe_allow_html=True)
                        st.markdown(f'<div class="digit-display">{format_digits_display(sequence)}</div>', 
                                   unsafe_allow_html=True)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.ds_digits_hidden = True
                        st.session_state.ds_showing_digits = False
                        # Don't start timer here - wait for recording to start
                        st.session_state.ds_answer_start_time = None
                        st.rerun()
            
            # Recording phase
            else:
                # CRITICAL: Double-check phase to prevent UI bleeding
                if st.session_state.current_phase == 4:
                    st.markdown('<p class="countdown-text">🎤 Say the digits BACKWARDS!</p>', unsafe_allow_html=True)
                    
                    # TIMER FIX: Start timer when entering recording phase (digits just disappeared)
                    # This captures when user could first start recording
                    if st.session_state.ds_recording_start_time is None:
                        st.session_state.ds_recording_start_time = time.time()
                    
                    audio = mic_recorder(
                        start_prompt="🔴 START RECORDING",
                        stop_prompt="⏹️ STOP RECORDING",
                        just_once=True,
                        key=f"ds_recorder_{level}_{st.session_state.ds_attempts}"
                    )
                else:
                    audio = None  # Not in Phase 4, skip mic_recorder

                if audio:
                    # TIMER FIX: Capture end time immediately when audio is received
                    st.session_state.ds_recording_end_time = time.time()
                    
                    # BUG 8 FIX: Check for valid audio
                    if audio.get("bytes") and len(audio["bytes"]) > 0:
                        with st.spinner("🔄 Transcribing..."):
                            transcript = transcribe_audio(audio["bytes"])
                            
                            if transcript.startswith("ERROR:"):
                                st.session_state.ds_error = transcript.replace("ERROR: ", "")
                                st.rerun()
                            
                            st.session_state.ds_transcript = transcript
                        
                        user_digits = parse_spoken_digits(transcript)
                        st.session_state.ds_user_answer = user_digits
                        
                        # BUG 3 FIX: Use tested_sequence for correct answer
                        correct_answer = list(reversed(st.session_state.ds_tested_sequence))
                        
                        # TIMER FIX: Calculate time from when recording phase started to when audio was captured
                        if st.session_state.ds_recording_start_time and st.session_state.ds_recording_end_time:
                            attempt_time = st.session_state.ds_recording_end_time - st.session_state.ds_recording_start_time
                            st.session_state.ds_total_time += attempt_time
                            st.session_state.ds_last_attempt_time = round(attempt_time, 1)
                        else:
                            # Fallback
                            st.session_state.ds_last_attempt_time = 0
                        
                        # Check answer
                        is_correct = user_digits == correct_answer
                        st.session_state.ds_last_result = is_correct
                        st.session_state.ds_total_attempts += 1
                        
                        # PER-LEVEL TIMING: Store attempt time for current level (overwrites on retry)
                        attempt_time = st.session_state.ds_last_attempt_time
                        if level == 1:
                            st.session_state.ds_level1_time = attempt_time
                        elif level == 2:
                            st.session_state.ds_level2_time = attempt_time
                        elif level == 3:
                            st.session_state.ds_level3_time = attempt_time
                        
                        if is_correct:
                            st.session_state.ds_max_achieved = num_digits
                            # Level 3 completed - immediately advance to Phase 5 (skip feedback to prevent UI bleeding)
                            if level >= 3:
                                st.session_state.ds_test_complete = True
                                st.session_state.current_phase = 5
                                st.rerun()  # Immediately go to Phase 5
                            else:
                                st.session_state.ds_level += 1
                                st.session_state.ds_attempts = 0
                                st.session_state.ds_current_sequence = generate_digit_sequence(num_digits + 1)
                                st.session_state.ds_tested_sequence = st.session_state.ds_current_sequence.copy()
                        else:
                            st.session_state.ds_attempts += 1
                            if st.session_state.ds_attempts >= 5:
                                st.session_state.ds_test_complete = True
                                # Failed all attempts - immediately advance to Phase 5
                                st.session_state.current_phase = 5
                                st.rerun()  # Immediately go to Phase 5
                            else:
                                # Generate new sequence for retry, but keep tested_sequence as the old one until next show
                                st.session_state.ds_current_sequence = generate_digit_sequence(num_digits)
                        
                        # TIMER FIX: Reset timer for next attempt
                        st.session_state.ds_recording_start_time = None
                        st.session_state.ds_recording_end_time = None
                        
                        st.session_state.ds_digits_hidden = False
                        st.rerun()
                    else:
                        st.session_state.ds_error = "No audio was captured. Please check your microphone."
                        st.rerun()
            
            # Feedback UI removed to prevent bleeding issues - data is still tracked for CSV/visualization
    
    else:
        # Test complete - AUTO-ADVANCE to Phase 5 (no results display here to prevent UI bleeding)
        # Results will be shown on the final results page (Phase 6)
        # DO NOT reset ds_test_complete - keep it True to maintain correct state
        st.session_state.current_phase = 5
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: ABSTRACT SYNTHESIS TEST
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_phase == 5:
    # NUCLEAR OPTION: Force clear ALL Phase 4 display state to prevent UI bleeding
    st.session_state.ds_last_result = None
    st.session_state.ds_transcript = ""
    st.session_state.ds_user_answer = []
    st.session_state.ds_showing_digits = False
    st.session_state.ds_digits_hidden = False
    
    st.markdown('<p class="phase-header">Phase 5: Abstract Synthesis Test</p>', unsafe_allow_html=True)
    
    # Use container to create rendering boundary - prevents Phase 4 remnants from bleeding through
    with st.container():
        # Fetch paper if needed
        if st.session_state.as_paper is None:
            with st.spinner("📚 Fetching random arxiv paper..."):
                paper = fetch_random_arxiv_paper()
                if paper and "error" not in paper:
                    st.session_state.as_paper = paper
                    # BUG 5 FIX: Auto-start reading timer immediately when paper loads
                    st.session_state.as_reading_started = True
                    st.session_state.as_reading_end = time.time() + 60
                else:
                    st.error("Cannot fetch paper. Retrying...")
                    time.sleep(2)
                    st.rerun()
            st.rerun()
        
        paper = st.session_state.as_paper

    
    # BUG 8 FIX: Show error and retry button if there was an error
    if st.session_state.as_error:
        st.error(f"❌ Error: {st.session_state.as_error}")
        if st.button("🔄 RETRY TEST", type="primary", use_container_width=True, key="as_retry"):
            # Reset Phase 4 state
            st.session_state.as_paper = None
            st.session_state.as_reading_started = False
            st.session_state.as_reading_end = None
            st.session_state.as_abstract_hidden = False
            st.session_state.as_transcript = ""
            st.session_state.as_score = None
            st.session_state.as_feedback = ""
            st.session_state.as_recording_complete = False
            st.session_state.as_explanation_started = False
            st.session_state.as_explanation_end = None
            st.session_state.as_error = None
            st.session_state.as_grading_error = None
            st.rerun()
    
    elif not st.session_state.as_recording_complete:
        # Reading phase
        if not st.session_state.as_abstract_hidden:
            st.subheader(f"📄 {paper['title']}")
            
            # Use st.empty() to prevent icon flashing during countdown
            timer_placeholder = st.empty()
            
            # BUG 5 FIX: Timer starts automatically - no button needed
            remaining = max(0, st.session_state.as_reading_end - time.time())
            
            if remaining > 0:
                timer_placeholder.markdown(f'<p class="big-timer">Reading: {int(remaining)}s</p>', unsafe_allow_html=True)
                st.info(paper['summary'])
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.as_abstract_hidden = True
                # BUG 6 FIX: Start explanation timer when abstract hides
                st.session_state.as_explanation_started = True
                st.session_state.as_explanation_end = time.time() + 60
                st.rerun()
        
        # Recording phase
        else:
            st.subheader(f"📄 {paper['title']}")
            st.warning("⚠️ Abstract is now hidden. Explain what you understood.")
            
            st.markdown("### 🎤 Record your explanation (60 seconds)")
            
            # Use st.empty() to prevent icon flashing during countdown
            explanation_timer_placeholder = st.empty()
            
            # BUG 6 FIX: Show countdown timer during explanation
            # Show countdown timer as visual guide (does NOT auto-advance)
            if st.session_state.as_explanation_end:
                remaining = max(0, st.session_state.as_explanation_end - time.time())
                if remaining > 0:
                    explanation_timer_placeholder.markdown(f'<p class="big-timer">{int(remaining)}s remaining</p>', unsafe_allow_html=True)
                else:
                    explanation_timer_placeholder.markdown('<p class="big-timer" style="color: #FF4444;">Time\'s up! Click STOP RECORDING.</p>', unsafe_allow_html=True)
            
            audio = mic_recorder(
                start_prompt="🔴 START EXPLANATION",
                stop_prompt="⏹️ STOP RECORDING",
                just_once=True,
                key="as_recorder"
            )
            
            if audio:
                # BUG 8 FIX: Check for valid audio
                if audio.get("bytes") and len(audio["bytes"]) > 0:
                    with st.spinner("🔄 Transcribing audio... (8-15 seconds)"):
                        transcript = transcribe_audio(audio["bytes"])
                        if not transcript.startswith("ERROR"):
                            st.session_state.as_transcript = transcript
                            st.session_state.as_recording_complete = True
                        else:
                            st.session_state.as_error = transcript.replace("ERROR: ", "")
                    st.rerun()
                else:
                    st.session_state.as_error = "No audio was captured. Please check your microphone."
                    st.rerun()
            
            # Auto-refresh for countdown timer (safe now that Phase 4 feedback is removed)
            if st.session_state.as_explanation_end:
                remaining = max(0, st.session_state.as_explanation_end - time.time())
                if remaining > 0:
                    time.sleep(1)
                    st.rerun()

    
    else:
        # Grading phase
        st.subheader("📝 Your Explanation:")
        st.info(st.session_state.as_transcript)
        
        # BUG 7 FIX: Show grading error with retry button
        if st.session_state.as_grading_error:
            st.error(f"❌ Grading Error: {st.session_state.as_grading_error}")
            if st.button("🔄 RETRY GRADING", type="primary", use_container_width=True, key="as_retry_grading"):
                st.session_state.as_score = None
                st.session_state.as_feedback = ""
                st.session_state.as_grading_error = None
                st.rerun()
        
        elif st.session_state.as_score is None and st.session_state.gemini_api_key:
            with st.spinner("🤖 Grading with Gemini..."):
                prompt = f"""You are grading a scientific abstract comprehension test. Grade on 1-10 scale.
10 = perfectly captures core concept, 1 = completely wrong or off-topic.

GRADING RUBRIC (use this exactly):
- 9-10: Identifies main finding, method, and implications accurately
- 7-8: Captures main idea, minor gaps
- 5-6: Partial understanding, missing key concepts
- 3-4: Vague or mostly incorrect
- 1-2: Completely off-topic

Original Abstract: {paper['summary']}

User Explanation: {st.session_state.as_transcript}

Return ONLY valid JSON (no markdown):
{{"score": N, "reasoning": "brief feedback"}}"""
                
                response = call_gemini(prompt, st.session_state.gemini_api_key)
                
                # BUG 7 FIX: Use robust JSON parsing
                result, error = parse_gemini_json(response)
                if result:
                    st.session_state.as_score = result.get("score", 0)
                    st.session_state.as_feedback = result.get("reasoning", "")
                    st.session_state.as_grading_success = True  # Track successful grading
                else:
                    # IMPROVED ERROR: Show full error message from Gemini
                    st.session_state.as_grading_error = error or "Failed to parse grading response"
            st.rerun()
        
        if st.session_state.as_score is not None:
            score = st.session_state.as_score
            score_class = "success" if score >= 7 else "failure" if score < 5 else ""
            st.markdown(f'<div class="score-display {score_class}">Score: {score}/10</div>', 
                       unsafe_allow_html=True)
            st.write(f"**Feedback:** {st.session_state.as_feedback}")
            
            # DEBUG OUTPUT: Show Gemini success message
            if st.session_state.get("as_grading_success"):
                st.success(f"✅ Graded by {GEMINI_MODEL}")
        
        st.divider()
        
        # Show original abstract for comparison
        with st.expander("📄 View Original Abstract"):
            st.write(paper['summary'])
        
        # BUG 8 FIX: Only show FINISH button if grading was successful
        if st.session_state.as_score is not None:
            if st.button("▶️ FINISH & VIEW RESULTS", type="primary", use_container_width=True):
                st.session_state.current_phase = 6
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: STROOP TEST (Batch Format)
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_phase == 2:
    st.markdown('<p class="phase-header">Phase 2: Stroop Test</p>', unsafe_allow_html=True)
    
    # Initialize grids if not done
    if st.session_state.stroop_congruent_grid is None:
        st.session_state.stroop_congruent_grid = generate_stroop_grid(congruent=True)
    if st.session_state.stroop_incongruent_grid is None:
        st.session_state.stroop_incongruent_grid = generate_stroop_grid(congruent=False)
    
    stage = st.session_state.stroop_stage
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INSTRUCTIONS SCREEN
    # ═══════════════════════════════════════════════════════════════════════════
    if stage == "instructions":
        st.markdown("""
        <div class="stroop-instructions">
            <h2 style="color: #4B9AFF; text-align: center;">🧠 STROOP TEST INSTRUCTIONS</h2>
            <br>
            <p style="font-size: 18px; color: #ddd;">You'll see 25 colored words twice.</p>
            <br>
            <p style="font-size: 18px; color: #44DD44;"><strong>TEST 1 (Easy):</strong> Words match their colors. Say the <em>colors</em> out loud as fast as you can.</p>
            <br>
            <p style="font-size: 18px; color: #FF6B6B;"><strong>TEST 2 (Hard):</strong> Words DON'T match their colors. Say the <em>INK COLOR</em> (not the word!) as fast as you can.</p>
            <br>
            <p style="font-size: 16px; color: #888;">• Be accurate but fast. Errors will slow you down.</p>
            <p style="font-size: 16px; color: #888;">• Read top → bottom, left → right.</p>
            <p style="font-size: 16px; color: #888;">• Time limit: 2 minutes per screen.</p>
            <br>
            <p style="font-size: 16px; color: #888;">• The timer will start automatically when you see the grid.</p>
            <p style="font-size: 16px; color: #888;">• Press STOP when finished.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 START TEST", type="primary", use_container_width=True):
            st.session_state.stroop_stage = "congruent"
            st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONGRUENT TEST (Easy - words match colors)
    # ═══════════════════════════════════════════════════════════════════════════
    elif stage == "congruent":
        # Auto-start timer when grid appears
        if not st.session_state.stroop_timer_running and st.session_state.stroop_timer_start is None:
            st.session_state.stroop_timer_start = time.time()
            st.session_state.stroop_timer_running = True
        
        # Render the grid (no header - grid is first thing displayed)
        st.markdown(render_stroop_grid(st.session_state.stroop_congruent_grid), unsafe_allow_html=True)
        
        # Timer display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            elapsed = time.time() - st.session_state.stroop_timer_start
            
            # Check for timeout (2 minutes = 120 seconds)
            if elapsed >= 120:
                st.session_state.stroop_congruent_time = 120.0
                st.session_state.stroop_congruent_timeout = True
                st.session_state.stroop_timer_running = False
                st.session_state.stroop_stage = "congruent_result"
                st.rerun()
            
            st.markdown(f'<p class="stroop-timer stroop-timer-running">⏱️ {elapsed:.1f}s</p>', unsafe_allow_html=True)
            
            if st.button("⏹️ STOP", type="primary", use_container_width=True):
                st.session_state.stroop_congruent_time = elapsed
                st.session_state.stroop_timer_running = False
                st.session_state.stroop_stage = "congruent_result"
                st.rerun()
            
            # Auto-refresh to update timer
            time.sleep(0.1)
            st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONGRUENT RESULT
    # ═══════════════════════════════════════════════════════════════════════════
    elif stage == "congruent_result":
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px;">
            <span style="font-size: 48px;">✅</span>
            <h2 style="color: #44DD44;">Test 1 Complete</h2>
            <p style="font-size: 36px; color: #4B9AFF; font-family: 'Courier New', monospace;">
                Your time: {st.session_state.stroop_congruent_time:.1f} seconds
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("▶️ Continue to Test 2", type="primary", use_container_width=True):
            st.session_state.stroop_stage = "incongruent"
            st.session_state.stroop_timer_running = False
            st.session_state.stroop_timer_start = None
            st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INCONGRUENT TEST (Hard - words don't match colors)
    # ═══════════════════════════════════════════════════════════════════════════
    elif stage == "incongruent":
        # Auto-start timer when grid appears
        if not st.session_state.stroop_timer_running and st.session_state.stroop_timer_start is None:
            st.session_state.stroop_timer_start = time.time()
            st.session_state.stroop_timer_running = True
        
        # Render the grid (no header - grid is first thing displayed)
        st.markdown(render_stroop_grid(st.session_state.stroop_incongruent_grid), unsafe_allow_html=True)
        
        # Timer display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            elapsed = time.time() - st.session_state.stroop_timer_start
            
            # Check for timeout (2 minutes = 120 seconds)
            if elapsed >= 120:
                st.session_state.stroop_incongruent_time = 120.0
                st.session_state.stroop_incongruent_timeout = True
                st.session_state.stroop_timer_running = False
                st.session_state.stroop_stage = "incongruent_result"
                st.rerun()
            
            st.markdown(f'<p class="stroop-timer stroop-timer-running">⏱️ {elapsed:.1f}s</p>', unsafe_allow_html=True)
            
            if st.button("⏹️ STOP", type="primary", use_container_width=True, key="stroop_stop_2"):
                st.session_state.stroop_incongruent_time = elapsed
                st.session_state.stroop_timer_running = False
                st.session_state.stroop_stage = "incongruent_result"
                st.rerun()
            
            # Auto-refresh to update timer
            time.sleep(0.1)
            st.rerun()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INCONGRUENT RESULT (Final Summary)
    # ═══════════════════════════════════════════════════════════════════════════
    elif stage == "incongruent_result":
        congruent_time = st.session_state.stroop_congruent_time or 0
        incongruent_time = st.session_state.stroop_incongruent_time or 0
        interference = incongruent_time - congruent_time
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px;">
            <span style="font-size: 48px;">✅</span>
            <h2 style="color: #44DD44;">Stroop Test Complete!</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #1a4d1a; border-radius: 10px;">
                <p style="color: #888; font-size: 14px;">Congruent (Easy)</p>
                <p style="font-size: 32px; color: #44DD44; font-family: 'Courier New', monospace;">{congruent_time:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #4d1a1a; border-radius: 10px;">
                <p style="color: #888; font-size: 14px;">Incongruent (Hard)</p>
                <p style="font-size: 32px; color: #FF6B6B; font-family: 'Courier New', monospace;">{incongruent_time:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            interference_color = "#FFD700" if interference > 0 else "#4B9AFF"
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #3d3d1a; border-radius: 10px;">
                <p style="color: #888; font-size: 14px;">Interference Effect</p>
                <p style="font-size: 32px; color: {interference_color}; font-family: 'Courier New', monospace;">{interference:.1f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("✅ NEXT TEST", type="primary", use_container_width=True):
            st.session_state.stroop_stage = "complete"
            st.session_state.current_phase = 3
            st.session_state.vf_letter = random.choice(string.ascii_uppercase)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: DATA LOGGING & RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.current_phase == 6:
    st.markdown('<p class="phase-header">Phase 6: Results & Data Logging</p>', unsafe_allow_html=True)
    
    # Compile all data
    # Calculate Stroop interference effect
    stroop_congruent = st.session_state.stroop_congruent_time or 0
    stroop_incongruent = st.session_state.stroop_incongruent_time or 0
    stroop_interference = stroop_incongruent - stroop_congruent
    
    data = {
        # Test timestamp (12-hour format with AM/PM)
        "test_timestamp": datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"),
        
        # Morning Vitals (Pre-Semax)
        "morning_bp": st.session_state.morning_bp,
        "morning_hr": st.session_state.morning_hr,
        "morning_spo2": st.session_state.morning_spo2,
        "sleep_total_hours": st.session_state.sleep_total,
        "sleep_rem_hours": st.session_state.sleep_rem,
        "sleep_deep_hours": st.session_state.sleep_deep,
        "morning_hrv_ms": st.session_state.morning_hrv,
        "morning_timestamp": st.session_state.morning_timestamp,
        
        # Current Vitals (Pre-Test)
        "current_bp": st.session_state.current_bp,
        "current_hr": st.session_state.current_hr,
        "current_spo2": st.session_state.current_spo2,
        "hours_since_semax": st.session_state.hours_since_semax,
        "total_dose_mcg": st.session_state.total_dose_mcg,
        "current_timestamp": st.session_state.current_timestamp,
        
        # Subjective Notes
        "subjective_notes": st.session_state.subjective_notes,
        
        # Cognitive Tests
        "letter_tested": st.session_state.vf_letter,
        "verbal_fluency_score": st.session_state.vf_score or 0,
        "verbal_fluency_transcript": st.session_state.vf_transcript,
        "max_digit_span": st.session_state.ds_max_achieved or 5,
        "digit_span_attempts": st.session_state.ds_total_attempts,
        "digit_span_time_seconds": round(st.session_state.ds_total_time, 1),
        # PER-LEVEL TIMING: Individual level times for CSV
        "digit_span_level1_time": round(st.session_state.ds_level1_time, 1),
        "digit_span_level2_time": round(st.session_state.ds_level2_time, 1),
        "digit_span_level3_time": round(st.session_state.ds_level3_time, 1),
        "arxiv_paper_id": st.session_state.as_paper.get("id", "") if st.session_state.as_paper else "",
        "arxiv_paper_title": st.session_state.as_paper.get("title", "") if st.session_state.as_paper else "",
        "synthesis_score": st.session_state.as_score or 0,
        "synthesis_transcript": st.session_state.as_transcript,
        "synthesis_feedback": st.session_state.as_feedback,
        "stroop_congruent_time": round(stroop_congruent, 1),
        "stroop_incongruent_time": round(stroop_incongruent, 1),
        "stroop_interference_effect": round(stroop_interference, 1),
    }
    
    # Save to CSV
    save_to_csv(data)
    
    st.success("✅ Data saved to semax_cognitive_log.csv")
    
    # Display summary
    st.subheader("📊 Test Summary")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VITALS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌅 Morning Vitals (Pre-Semax)")
        st.write(f"🩸 BP: {data['morning_bp']}")
        st.write(f"❤️ HR: {data['morning_hr']} BPM")
        st.write(f"💨 SpO2: {data['morning_spo2']}%")
        st.write(f"💓 HRV: {data['morning_hrv_ms']} ms")
        st.write(f"⏰ Time: {data['morning_timestamp']}")
    
    with col2:
        st.markdown("### 🧪 Current Vitals (Pre-Test)")
        st.write(f"🩸 BP: {data['current_bp']}")
        st.write(f"❤️ HR: {data['current_hr']} BPM")
        st.write(f"💨 SpO2: {data['current_spo2']}%")
        st.write(f"💊 Hours Since Semax: {data['hours_since_semax']}")
        st.write(f"💉 Total Dose: {data['total_dose_mcg']} mcg")
        st.write(f"⏰ Time: {data['current_timestamp']}")
    
    # Sleep Data - VERTICAL layout
    st.markdown("### 😴 Sleep Data")
    st.write(f"Total Sleep: {data['sleep_total_hours']} hours")
    st.write(f"REM Sleep: {data['sleep_rem_hours']} hours")
    st.write(f"Deep Sleep: {data['sleep_deep_hours']} hours")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COGNITIVE TESTS DISPLAY - Balanced 2x2 layout
    # ═══════════════════════════════════════════════════════════════════════════
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🗣️ Verbal Fluency")
        st.write(f"Letter: {data['letter_tested']}")
        st.write(f"Score: {data['verbal_fluency_score']} words")
        
        st.markdown("### 🔢 Digit Span")
        st.write(f"Max Achieved: {data['max_digit_span']} digits")
        st.write(f"Total Attempts: {data['digit_span_attempts']}")
        st.write(f"Total Time: {data['digit_span_time_seconds']}s")

    
    with col2:
        st.markdown("### 🎨 Stroop Test")
        st.write(f"Congruent: {data['stroop_congruent_time']}s")
        st.write(f"Incongruent: {data['stroop_incongruent_time']}s")
        st.write(f"Interference: {data['stroop_interference_effect']}s")
        
        st.markdown("### � Abstract Synthesis")
        st.write(f"Score: {data['synthesis_score']}/10")
    
    st.divider()
    
    # Show CSV data
    with st.expander("📁 View All Logged Data"):
        if os.path.exists("semax_cognitive_log.csv"):
            df = pd.read_csv("semax_cognitive_log.csv")
            st.dataframe(df)
    
    # Render data visualizations (only shows if 2+ days of data)
    # Render data visualizations (only shows if 2+ days of data)
    render_data_visualizations()
    
    st.info("🔄 To run another test, please refresh the page.")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption("Semax Cognitive Testing Suite • Cognitive Performance Tracking")
