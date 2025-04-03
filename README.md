# YouTube Video Summarizer API

A Flask-based web service that extracts and summarizes English subtitles when you enter a YouTube URL.

## Features

- Extracts English subtitles from YouTube videos (both manual and auto-generated)
- Processes long transcripts with token limit optimization
- Generates concise bullet-point summaries using Deepseek's AI
- Handles various YouTube URL formats
- Comprehensive error logging and handling

## Prerequisites

- Python 3.8+
- Deepseek API key
- YouTube video url

## Installation

1. Clone the repository:
```bash
git clone https:///github.com/yeong7513/video-summarizer.git
cd video-summarizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```env
DEEPSEEK_API_KEY=your_api_key_here
```

## Usage

### Running the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

### API Endpoint

**POST** `/summarize`

Request body:
```json
{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

Successful response:
```json
{
    "summary": "Bullet-point summary of the video...",
    "video_id": "VIDEO_ID"
}
```

Error responses:
- 400: Invalid request
- 404: Subtitles not available
- 500: Server error

## Configuration

Environment variables:
- `DEEPSEEK_API_KEY`: Required - Your Deepseek API key
- `MAX_TOKENS`: Optional - Max tokens for summary (default: 300)
- `MAX_INPUT_TOKENS`: Optional - Max input tokens (default: 6000)

## Code Structure

- `extract_video_id(url)`: Extracts YouTube video ID from URL
- `get_transcript(video_id)`: Retrieves English subtitles
- `truncate_text(text, max_tokens)`: Ensures text fits token limit
- `summarize_with_deepseek(text)`: Generates summary using Deepseek API
- `/summarize` endpoint: Main API endpoint

## Error Handling

The service logs all errors with detailed tracebacks. Common errors include:
- Invalid YouTube URLs
- Videos without English subtitles
- API quota limits
- Token limit exceeded

## Limitations

- Currently only supports English subtitles
- Maximum input length limited by Deepseek's token limits
- Requires stable internet connection

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
