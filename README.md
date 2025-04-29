# YouTube Insight Assistant

An advanced application for analyzing, understanding, and interacting with YouTube video content using AI.

## Features

- **Video Analysis**: Extract and analyze video metadata and transcripts
- **Content Insights**: Identify key topics, important segments, and generate summaries
- **Interactive Chat**: Ask questions about the video content and get AI-powered answers
- **Visual Data**: Visualize content patterns and speaking pace

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd Youtube_Chatbot
```

2. Create a virtual environment:
```
conda create -n youtube_insight python=3.9
conda activate youtube_insight
```

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter a YouTube URL in the input field and explore the different features:
   - Video Analysis: Get detailed metadata and transcript analysis
   - Content Insights: View key topics and generate summaries
   - Chat: Ask questions about the video content

## Requirements

- Python 3.9+
- Groq API key (for LLM access)
- Internet connection (for accessing YouTube videos)

## Technical Details

This application uses:
- **Streamlit**: For the web interface
- **LangChain**: For orchestrating the AI components
- **Groq**: For large language model inference
- **FAISS**: For efficient similarity search
- **HuggingFace**: For embeddings
- **YouTube Transcript API**: For extracting video transcripts
- **PyTube**: For fetching video metadata
- **Pandas/Matplotlib/Seaborn**: For data visualization

## Future Improvements

- Multi-language support
- Sentiment analysis of video content
- Speaker identification in multi-person videos
- Integration with YouTube comments
- Comparison between multiple videos

## License

[MIT License](LICENSE)

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the LLM framework
- [Streamlit](https://streamlit.io/) for the web app framework
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction