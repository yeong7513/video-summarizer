from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import logging
import torch
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Инициализация модели для английского языка
tokenizer_en = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model_en = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

@app.route('/', methods=['GET'])
def home():
    return "YouTube Video Summarizer (English)"

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
    """Получение английских субтитров"""
    try:
        # Попытка получить ручные английские субтитры
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        logger.info(f"transcript: {transcript}")
        return " ".join([entry['text'] for entry in transcript])
    
    except (TranscriptsDisabled, NoTranscriptFound):
        try:
            # Попытка получить автоматические английские субтитры
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            logger.info(f"transcript_list: {transcript_list}")
            auto_sub = transcript_list.find_generated_transcript(['en'])
            logger.info(f"auto_sub: {auto_sub}")
            transcript = auto_sub.fetch()
            logger.info(f"transcript(auto_sub): {transcript}")
            return " ".join([entry.text for entry in transcript])
        
        except Exception as e:
            logger.error(f"Ошибка получения субтитров: {str(e)}")
            raise Exception("Английские субтитры недоступны")

def summarize_text(text):
    """Суммаризация английского текста"""
    inputs = tokenizer_en(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )
    
    summary_ids = model_en.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    return tokenizer_en.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "Missing URL"}), 400

        video_id = extract_video_id(data['url'])
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        text = get_transcript(video_id)
        logger.info(f"text: {text}")
        summary = summarize_text(text)
        logger.info(f"summary: {summary}")
        
        return jsonify({
            "summary": summary,
            "video_id": video_id
        })
    
    except Exception as e:
        logger.error(f"Processing error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)