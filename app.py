import os
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from urllib.parse import urlparse, parse_qs
import time
import json
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytube import YouTube
import re

# Load environment variables
load_dotenv()

# Initialize Streamlit interface with custom theme
st.set_page_config(
    page_title="YouTube Insight Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved text visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #CC0000;  /* Darker red for better contrast */
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #404040;  /* Darker gray for better visibility */
        margin-bottom: 1rem;
    }
    .stApp {
        background-color: #f9f9f9;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #000000;  /* Ensuring text is black for visibility */
    }
    .chat-message.user {
        background-color: #e6f2ff;  /* Lighter blue background */
        border-left: 5px solid #0066cc;  /* Darker blue border */
    }
    .chat-message.bot {
        background-color: #e6ffe6;  /* Lighter green background */
        border-left: 5px solid #009933;  /* Darker green border */
    }
    .video-stats {
        background-color: #fff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        color: #000000;  /* Ensuring text is black */
    }
    /* YouTube-style buttons */
    .stButton button {
        background-color: #065FD4;  /* YouTube blue */
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0356C2;  /* Darker blue on hover */
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    /* Special styling for primary action buttons */
    .stButton.primary button {
        background-color: #CC0000;  /* YouTube red for primary actions */
    }
    .stButton.primary button:hover {
        background-color: #990000;  /* Darker red on hover */
    }
    /* Ensure all text has good contrast */
    p, h1, h2, h3, h4, h5, h6, li, span, div {
        color: #333333;  /* Dark gray for most text */
    }
    /* Override Streamlit's default styles for better visibility */
    .stMarkdown, .stText {
        color: #333333 !important;
    }
    /* Make links more visible */
    a {
        color: #0066cc !important;
        text-decoration: underline;
    }
    /* Improve visibility in expandable sections */
    .streamlit-expanderContent {
        background-color: #ffffff;
        color: #333333;
        padding: 1rem;
        border-radius: 0.5rem;
    }

    /* Style the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f0f0;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #065FD4 !important;
        color: white !important;
    }

    /* Style metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
        color: #065FD4;
    }

    /* Style the sidebar */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
        border-right: 1px solid #e0e0e0;
    }

    /* Style the main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Style the video container */
    [data-testid="stVideo"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }

    /* Style the expanders */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #333333;
        background-color: #f9f9f9;
        border-radius: 4px;
    }

    /* Style code blocks */
    code {
        background-color: #f0f0f0;
        padding: 0.2em 0.4em;
        border-radius: 3px;
        font-size: 85%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'video_cache' not in st.session_state:
    st.session_state['video_cache'] = {}

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'video_metadata' not in st.session_state:
    st.session_state['video_metadata'] = None

if 'transcript_analysis' not in st.session_state:
    st.session_state['transcript_analysis'] = None

st.markdown("<h1 class='main-header'>🎥 YouTube Insight Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze, understand, and interact with any YouTube video content</p>", unsafe_allow_html=True)

def extract_video_id(url):
    """Extract the video ID from a YouTube URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    return None

def get_transcript(video_id):
    """Get transcript of the YouTube video with timestamps"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Keep the timestamps for advanced analysis
        return transcript_list, ' '.join([item['text'] for item in transcript_list])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None, None

def get_video_metadata(video_id):
    """Fetch metadata about the YouTube video using pytube"""
    try:
        # Initialize YouTube object
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")

        # Extract metadata
        metadata = {
            "title": yt.title,
            "channel": yt.author,
            "views": yt.views,
            "publish_date": yt.publish_date.strftime("%Y-%m-%d") if yt.publish_date else "Unknown",
            "length": yt.length,
            "description": yt.description,
            "thumbnail_url": yt.thumbnail_url,
            "keywords": yt.keywords if hasattr(yt, 'keywords') else [],
            "rating": yt.rating if hasattr(yt, 'rating') else None
        }
        return metadata
    except Exception as e:
        st.warning(f"Could not fetch complete video metadata: {str(e)}")
        return None

def analyze_transcript(transcript_list):
    """Perform basic analysis on the transcript"""
    if not transcript_list:
        return None

    # Extract data for analysis
    durations = [item['duration'] for item in transcript_list]
    text_lengths = [len(item['text']) for item in transcript_list]

    # Calculate statistics
    total_duration = sum(durations)
    avg_segment_length = sum(text_lengths) / len(text_lengths)
    word_count = sum(len(item['text'].split()) for item in transcript_list)

    # Find segments with highest information density (length/duration)
    info_density = [(i, len(item['text'].split()) / item['duration'] if item['duration'] > 0 else 0)
                    for i, item in enumerate(transcript_list)]
    info_density.sort(key=lambda x: x[1], reverse=True)
    key_segments = [transcript_list[idx] for idx, _ in info_density[:5] if _ > 0]

    # Identify potential topics using simple keyword frequency
    all_text = ' '.join([item['text'] for item in transcript_list])
    words = re.findall(r'\b\w+\b', all_text.lower())
    # Filter out common stop words (simplified)
    stop_words = {'the', 'and', 'is', 'in', 'to', 'a', 'of', 'that', 'for', 'it', 'with', 'as', 'this', 'on', 'are'}
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

    # Count word frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Get top keywords
    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_duration": total_duration,
        "segment_count": len(transcript_list),
        "avg_segment_length": avg_segment_length,
        "word_count": word_count,
        "key_segments": key_segments,
        "top_keywords": top_keywords
    }

def format_time(seconds):
    """Format seconds into MM:SS format"""
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_word_cloud(keywords):
    """Create a word cloud from keywords"""
    if not keywords:
        return None

    # Create a DataFrame for the word cloud
    df = pd.DataFrame(keywords, columns=['word', 'count'])

    # Create a simple bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x='count', y='word', data=df.sort_values('count', ascending=False).head(10))
    plt.title('Top Keywords in Video')
    plt.tight_layout()

    return plt

def setup_qa_chain(transcript):
    """Set up the QA chain with the transcript"""
    # Text splitting with larger chunks for better performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Larger chunks mean fewer embeddings to process
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(transcript)

    # Show progress information
    progress_text = st.empty()
    progress_text.info(f"Processing {len(chunks)} text chunks...")

    # Create embeddings and vector store with a lighter model
    progress_text.info("Loading embedding model (this is usually the slowest step)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    progress_text.info("Creating vector store...")
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Retrieve more context
    progress_text.empty()

    # Create enhanced prompt template with more context
    prompt = PromptTemplate.from_template("""
    You are an AI assistant specialized in analyzing YouTube video content.
    Answer the question based on the following transcript context from the video.

    Context: {context}

    Question: {question}

    Provide a detailed and informative answer. If the information is not available in the context,
    say so clearly rather than making up information. If appropriate, include timestamps or
    reference specific parts of the video.

    Answer:""")

    # Initialize LLM with a currently supported model
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192",  # Using a currently supported model
        temperature=0.3  # Lower temperature for more factual responses
    )

    # Create chain with memory
    chain = RunnableParallel(
        {
            'context': retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            'question': RunnablePassthrough()
        }
    ) | prompt | llm

    return chain

# Create sidebar for navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e1/Logo_of_YouTube_%282015-2017%29.svg", width=200)
nav_option = st.sidebar.radio("Navigation", ["Home", "Video Analysis", "Chat", "About"])

# Main content area
if nav_option == "Home":
    st.markdown("""
    ## Welcome to YouTube Insight Assistant

    This application helps you analyze and interact with YouTube video content in several ways:

    1. **Video Analysis** - Get detailed insights about any YouTube video including metadata,
       transcript analysis, and key content highlights

    2. **Interactive Chat** - Ask questions about the video content and get AI-powered answers
       based on the video transcript

    3. **Content Summarization** - Get summaries of video content at different levels of detail

    To get started, enter a YouTube URL below or navigate to the Video Analysis section.
    """)

    # URL input on home page
    youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.session_state['current_video_id'] = video_id
            st.success(f"Video ID: {video_id} - Now go to Video Analysis tab to analyze this video")
            # Auto-switch to analysis tab
            st.rerun()
        else:
            st.error("Invalid YouTube URL")

elif nav_option == "Video Analysis":
    # Get video ID from URL or session state
    video_id = None
    youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.session_state['current_video_id'] = video_id
    elif 'current_video_id' in st.session_state:
        video_id = st.session_state['current_video_id']

    if video_id:
        # Display video
        st.video(f"https://www.youtube.com/watch?v={video_id}")

        # Create tabs for different analysis views
        tabs = st.tabs(["Overview", "Transcript Analysis", "Content Insights", "Visual Data"])

        # Process video if not already processed
        if video_id not in st.session_state['video_cache'] or st.button("Refresh Analysis"):
            with st.spinner("Processing video... This may take a moment."):
                # Get video metadata
                metadata = get_video_metadata(video_id)
                st.session_state['video_metadata'] = metadata

                # Get transcript
                transcript_list, transcript_text = get_transcript(video_id)

                if transcript_text:
                    # Analyze transcript
                    analysis = analyze_transcript(transcript_list)
                    st.session_state['transcript_analysis'] = analysis

                    # Setup QA chain
                    start_time = time.time()
                    qa_chain = setup_qa_chain(transcript_text)
                    processing_time = time.time() - start_time

                    # Store in session state and cache
                    st.session_state['qa_chain'] = qa_chain
                    st.session_state['video_cache'][video_id] = qa_chain
                    st.session_state['transcript_text'] = transcript_text
                    st.session_state['transcript_list'] = transcript_list

                    st.success(f"Processing completed in {processing_time:.2f} seconds")

        # Overview tab
        with tabs[0]:
            if 'video_metadata' in st.session_state and st.session_state['video_metadata']:
                metadata = st.session_state['video_metadata']

                # Display metadata in a nice format
                col1, col2 = st.columns([1, 2])

                with col1:
                    if 'thumbnail_url' in metadata and metadata['thumbnail_url']:
                        st.image(metadata['thumbnail_url'], use_column_width=True)

                with col2:
                    st.markdown(f"### {metadata['title']}")
                    st.markdown(f"**Channel:** {metadata['channel']}")
                    st.markdown(f"**Published:** {metadata['publish_date']}")
                    st.markdown(f"**Views:** {metadata['views']:,}")
                    st.markdown(f"**Duration:** {format_time(metadata['length'])}")

                # Description in expandable section
                with st.expander("Video Description"):
                    st.markdown(metadata['description'])

                # Keywords
                if 'keywords' in metadata and metadata['keywords']:
                    st.markdown("### Keywords")
                    st.write(", ".join(metadata['keywords']))
            else:
                st.info("No metadata available for this video.")

        # Transcript Analysis tab
        with tabs[1]:
            if 'transcript_analysis' in st.session_state and st.session_state['transcript_analysis']:
                analysis = st.session_state['transcript_analysis']

                # Display analysis stats
                st.markdown("### Transcript Statistics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Duration", f"{format_time(analysis['total_duration'])}")
                col2.metric("Word Count", f"{analysis['word_count']:,}")
                col3.metric("Segments", f"{analysis['segment_count']:,}")

                # Display key segments
                st.markdown("### Key Segments")
                for i, segment in enumerate(analysis['key_segments']):
                    with st.expander(f"Segment {i+1} - {format_time(segment['start'])}"):
                        st.markdown(f"**Text:** {segment['text']}")
                        st.markdown(f"**Time:** {format_time(segment['start'])} - {format_time(segment['start'] + segment['duration'])}")
                        st.markdown(f"**Duration:** {segment['duration']:.2f} seconds")

                # Display full transcript
                with st.expander("Full Transcript"):
                    if 'transcript_list' in st.session_state:
                        for segment in st.session_state['transcript_list']:
                            st.markdown(f"**[{format_time(segment['start'])}]** {segment['text']}")
            else:
                st.info("No transcript analysis available for this video.")

        # Content Insights tab
        with tabs[2]:
            if 'transcript_analysis' in st.session_state and st.session_state['transcript_analysis']:
                analysis = st.session_state['transcript_analysis']

                # Display top keywords
                st.markdown("### Top Keywords")
                keyword_data = pd.DataFrame(analysis['top_keywords'], columns=['Keyword', 'Frequency'])
                st.bar_chart(keyword_data.set_index('Keyword'))

                # Generate a summary using the QA chain
                if 'qa_chain' in st.session_state:
                    st.markdown("### AI-Generated Summary")

                    summary_length = st.radio("Summary Length", ["Brief", "Detailed", "Comprehensive"], horizontal=True)

                    if st.button("Generate Summary"):
                        with st.spinner("Generating summary..."):
                            try:
                                if summary_length == "Brief":
                                    question = "Provide a brief 2-3 sentence summary of this video."
                                elif summary_length == "Detailed":
                                    question = "Provide a detailed paragraph summary of this video covering the main points."
                                else:
                                    question = "Provide a comprehensive summary of this video with all key points, examples, and conclusions."

                                response = st.session_state['qa_chain'].invoke(question)
                                st.markdown(response.content)
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
            else:
                st.info("No content insights available for this video.")

        # Visual Data tab
        with tabs[3]:
            if 'transcript_list' in st.session_state and st.session_state['transcript_list']:
                st.markdown("### Visual Data Analysis")

                # Create a speaking pace analysis
                if 'transcript_analysis' in st.session_state and st.session_state['transcript_analysis']:
                    analysis = st.session_state['transcript_analysis']

                    # Display top keywords as a bar chart
                    st.markdown("### Keyword Frequency")
                    keyword_df = pd.DataFrame(analysis['top_keywords'], columns=['word', 'count'])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='count', y='word', data=keyword_df, ax=ax)
                    plt.title('Top Keywords in Video')
                    st.pyplot(fig)

                    # Display speaking pace over time
                    st.markdown("### Speaking Pace Over Time")
                    transcript_list = st.session_state['transcript_list']

                    # Calculate words per minute for each segment
                    pace_data = []
                    for item in transcript_list:
                        if item['duration'] > 0:
                            words = len(item['text'].split())
                            wpm = (words / item['duration']) * 60
                            pace_data.append({
                                'time': item['start'],
                                'wpm': min(wpm, 300)  # Cap at 300 WPM for better visualization
                            })

                    if pace_data:
                        pace_df = pd.DataFrame(pace_data)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.lineplot(x='time', y='wpm', data=pace_df, ax=ax)
                        plt.title('Speaking Pace Throughout Video (Words Per Minute)')
                        plt.xlabel('Time (seconds)')
                        plt.ylabel('Words Per Minute')
                        st.pyplot(fig)
            else:
                st.info("No visual data available for this video.")

elif nav_option == "Chat":
    # Get video ID from session state
    if 'current_video_id' not in st.session_state:
        st.warning("Please select a video first in the Video Analysis section.")
    else:
        video_id = st.session_state['current_video_id']

        # Display video
        st.video(f"https://www.youtube.com/watch?v={video_id}")

        # Display chat history
        st.markdown("### Chat with the Video")

        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.markdown(f"<div class='chat-message user'><b>You:</b> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message bot'><b>AI:</b> {message['content']}</div>", unsafe_allow_html=True)

        # Question input
        question = st.text_input("Your question:", placeholder="What is this video about?")

        if question and 'qa_chain' in st.session_state:
            if st.button("Send"):
                # Add user message to chat history
                st.session_state['chat_history'].append({
                    'role': 'user',
                    'content': question
                })

                with st.spinner("Thinking..."):
                    try:
                        # Get answer
                        response = st.session_state['qa_chain'].invoke(question)

                        # Add AI response to chat history
                        st.session_state['chat_history'].append({
                            'role': 'assistant',
                            'content': response.content
                        })

                        # Rerun to update the chat display
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state['chat_history'] = []
            st.rerun()

else:  # About section
    st.markdown("""
    ## About YouTube Insight Assistant

    This application was developed to help users extract insights and interact with YouTube video content.

    ### Features

    - **Video Analysis**: Extract and analyze video metadata and transcripts
    - **Content Insights**: Identify key topics, important segments, and generate summaries
    - **Interactive Chat**: Ask questions about the video content and get AI-powered answers
    - **Visual Data**: Visualize content patterns and speaking pace

    ### Technologies Used

    - **Streamlit**: For the web interface
    - **LangChain**: For orchestrating the AI components
    - **Groq**: For large language model inference
    - **FAISS**: For efficient similarity search
    - **HuggingFace**: For embeddings
    - **YouTube Transcript API**: For extracting video transcripts
    - **PyTube**: For fetching video metadata

    ### Future Improvements

    - Multi-language support
    - Sentiment analysis of video content
    - Speaker identification in multi-person videos
    - Integration with YouTube comments
    - Comparison between multiple videos

    ### Feedback

    If you have any feedback or suggestions, please reach out to the developer.
    """)
