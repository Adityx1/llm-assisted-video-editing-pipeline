GEMINI_PROMPT_TEMPLATE = """
You are an expert video editor and content curator. I will provide you with a JSON transcript of a video.
Your task:
1. Identify a compelling Title for the video based on the context. 
2. Detect the Language (one of: Hindi, English, Hinglish).
3. Extract as many as possible relevant Keywords.
4. Select segments that comprise the "meaningful content" related to the core topic.
5. CRITICAL: Remove all fillers, silence mentions, long unrelated personal stories (if they don't add value), and any gibberish.
6. Ensure the final selected segments flow logically.
7. Targeted length: The final video should ideally be 15-20 minutes in total duration.

TRANSCRIPT JSON:
{transcript_json}

OUTPUT FORMAT (STRICT JSON):
{{
  "title": "...",
  "detected_language": "...",
  "detected_keywords": ["...", "..."],
  "meaningful_segments": [
    {{"start": 0.0, "end": 10.5, "text": "..."}},
    ...
  ]
}}
"""
