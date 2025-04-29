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

# Load environment variables
load_dotenv()

# Initialize Streamlit interface
st.set_page_config(
    page_title="YouTube Video Chatbot",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state for caching
if 'video_cache' not in st.session_state:
    st.session_state['video_cache'] = {}

st.title("💬 YouTube Video Chatbot")
st.caption("Ask questions about any YouTube video!")

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
    """Get transcript of the YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

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
    retriever = vector_store.as_retriever()
    progress_text.empty()

    # Create prompt template
    prompt = PromptTemplate.from_template("""
    Answer the question based on the following context:
    Context: {context}
    Question: {question}
    Answer:""")

    # Initialize LLM with a currently supported model
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-8b-8192"  # Using a currently supported model
    )

    # Create chain
    chain = RunnableParallel(
        {
            'context': retriever | RunnableLambda(lambda docs: "\n\n".join(doc.page_content for doc in docs)),
            'question': RunnablePassthrough()
        }
    ) | prompt | llm

    return chain

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # URL input
    youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        if video_id:
            # Display video
            st.video(youtube_url)

            # Check if video is already in cache
            if video_id in st.session_state['video_cache']:
                st.success("Using cached transcript and QA chain!")
                st.session_state['qa_chain'] = st.session_state['video_cache'][video_id]
            else:
                # Get transcript and setup chain
                with st.spinner("Processing video transcript... This may take a moment."):
                    transcript = get_transcript(video_id)
                    if transcript:
                        st.success("Transcript loaded successfully!")
                        start_time = time.time()
                        qa_chain = setup_qa_chain(transcript)
                        processing_time = time.time() - start_time
                        st.info(f"Processing completed in {processing_time:.2f} seconds")

                        # Store in session state and cache
                        st.session_state['qa_chain'] = qa_chain
                        st.session_state['video_cache'][video_id] = qa_chain
        else:
            st.error("Invalid YouTube URL")

with col2:
    # Question input
    st.subheader("Ask a Question")
    question = st.text_input("Your question:", placeholder="What is this video about?")

    if question and 'qa_chain' in st.session_state:
        with st.spinner("Thinking..."):
            try:
                # Get answer
                response = st.session_state['qa_chain'].invoke(question)

                # Display answer
                st.markdown("### Answer:")
                st.write(response.content)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Add some helpful information
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Paste a YouTube video URL in the input field
    2. Wait for the video to load and transcript to be processed
    3. Ask any question about the video content
    4. Get AI-powered answers based on the video transcript

    Note: The video must have English subtitles/transcript available.
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and LangChain")