import os, re, logging, traceback, tiktoken

from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Конфигурация Deepseek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MAX_TOKENS = 300  # Максимальное количество токенов для ответа
MAX_INPUT_TOKENS = 6000  # Максимальное количество токенов во входных данных

# Инициализация клиента и токенизатора
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
tokenizer = tiktoken.get_encoding("cl100k_base")

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Получение и обработка субтитров"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([entry['text'] for entry in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            auto_sub = transcript_list.find_generated_transcript(['en'])
            return " ".join([entry.text for entry in auto_sub.fetch()])
        except Exception as e:
            logger.error(f"Transcript error: {str(e)}")
            raise Exception("Subtitles unavailable")

def truncate_text(text, max_tokens):
    """Обрезка текста до максимального количества токенов"""
    tokens = tokenizer.encode(text)[:max_tokens]
    return tokenizer.decode(tokens)

def summarize_with_deepseek(text):
    """Суммаризация с обработкой ошибок и ограничениями"""
    try:
        truncated_text = truncate_text(text, MAX_INPUT_TOKENS)
        logger.info(f"Truncated text length: {len(tokenizer.encode(truncated_text))} tokens")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert summarizer. Create concise bullet-point summary in English."
                },
                {
                    "role": "user",
                    "content": f"Summarize this video transcript:\n\n{truncated_text}"
                }
            ],
            temperature=0.3,
            max_tokens=MAX_TOKENS
        )
        
        if not response.choices:
            raise APIError("Empty response from API")
            
        return response.choices[0].message.content
    
    except APIError as e:
        logger.error(f"Deepseek API Error: {e}")
        raise Exception(f"API Error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise Exception("Summary generation failed")

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing URL"}), 400

        video_id = extract_video_id(data['url'])
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        transcript_text = get_transcript(video_id)
        logger.info(f"Original transcript length: {len(transcript_text)} characters")
        
        summary = summarize_with_deepseek(transcript_text)
        return jsonify({"summary": summary, "video_id": video_id})
    
    except Exception as e:
        logger.error(f"Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY not found in environment variables")
        exit(1)
        
    app.run(host='0.0.0.0', port=5000)