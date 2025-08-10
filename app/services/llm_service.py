# app/services/llm_service.py
import google.generativeai as genai
import re
import time
import logging
from typing import List
from langdetect import detect
from app.config.settings import (
    GEMINI_API_KEYS, GEMINI_MODEL, LLM_TEMPERATURE, 
    LLM_MAX_TOKENS, LLM_TOP_P, LLM_TOP_K,
    gemini_request_count
)
from app.utils.auth_utils import get_current_gemini_key
from app.utils.text_utils import words_to_numbers, extract_key_terms, is_yes_no_question

logger = logging.getLogger(__name__)

def build_prompt(question: str, context_chunks: List[str]) -> str:
    """Build a comprehensive prompt for the LLM"""
    context = "\n\n---\n\n".join(context_chunks[:15])  # Focused context for reasoning
    
    is_yes_no = is_yes_no_question(question)
    key_terms = extract_key_terms(question.lower())
    
    hints = ""
    if key_terms:
        hints = f"""
**KEY TERMS TO LOOK FOR:** {", ".join(key_terms[:15])}
(The above terms from your question should help identify relevant information in the context below)
"""
    
    # Language detection
    try:
        lan = detect(question)
        has_non_english = lan != "en"
        language_instruction = "**IMPORTANT: Respond in the SAME LANGUAGE as the question.**\n\n" if has_non_english else ""
    except:
        language_instruction = ""

    if is_yes_no:
        response_instruction = """This is a yes/no question. Start your response with "Yes" or "No", then provide a brief explanation with specific details in 25-40 words total."""
    else:
        response_instruction = """Answer in ONE concise, direct paragraph (40-80 words maximum). Include specific numbers, amounts, percentages, and conditions. Use figures (1, 2, 3) instead of words (one, two, three) for all numbers."""

    return f"""{language_instruction}You are an intelligent insurance analyst who understands policy documents and can reason about their content. Follow this answer hierarchy:

**ANSWER HIERARCHY (in order of priority):**
1. **DIRECT ANSWER**: If the exact answer is clearly stated in the context, provide it directly with specific references
2. **DEDUCED ANSWER**: If not directly stated, deduce the answer by connecting related information from the context
3. **INTELLIGENT REASONING**: If neither direct nor deduction is possible, provide a well-reasoned answer based on general insurance knowledge and principles

**YOUR TASK:** {response_instruction}

**REASONING APPROACH:**
1. First, search for DIRECT answers in the context - look for exact matches or explicit statements
2. If no direct answer, DEDUCE by finding patterns, rules, and logical connections in the context data
3. Look for mathematical patterns, sequences, or operations demonstrated in similar examples
4. Only use general knowledge if absolutely no patterns can be deduced from the context
5. Always be transparent about your reasoning: "According to the document...", "Following the pattern shown...", "Based on similar examples..."
6. Include specific numbers, dates, amounts, and conditions when available
7. PRIORITY: Pattern deduction from context over standard knowledge

**CONTEXT TO ANALYZE:**
{context}

**QUESTION:** {question}

**ANSWER (follow the hierarchy - direct → deduced → intelligent reasoning):**"""

async def ask_llm(prompt: str, retry_count: int = 0) -> str:
    """Ask the LLM with retry logic"""
    try:
        current_api_key = get_current_gemini_key()
        logger.info(f"🔑 Using Gemini API KEY_{GEMINI_API_KEYS.index(current_api_key) + 1} (Request #{gemini_request_count}/12)")
        
        start_time = time.time()
        
        genai.configure(api_key=current_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
                top_p=LLM_TOP_P,
                top_k=LLM_TOP_K,
            )
        )
        
        logger.info(f"Gemini API call completed in {time.time() - start_time:.2f}s")
        
        result_text = ""
        if hasattr(response, 'text') and response.text:
            result_text = response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content.parts:
                result_text = candidate.content.parts[0].text.strip()
        
        if result_text:
            result_text = words_to_numbers(result_text)
            result_text = re.sub(r'^(Answer:|ANSWER:|Response:|Based on the context:)\s*', '', result_text, flags=re.IGNORECASE)
            result_text = re.sub(r'\s+', ' ', result_text).strip()
            if result_text.endswith('.') and not result_text.endswith('etc.') and not result_text.endswith('vs.'):
                result_text = result_text[:-1]
            
            return result_text
        
        return "No valid response generated."
            
    except Exception as e:
        logger.error(f"Gemini API error (attempt {retry_count + 1}): {str(e)}")
        
        if retry_count < 1 and ("quota" not in str(e).lower() and "limit" not in str(e).lower()):
            logger.info("Retrying with next API key...")
            return await ask_llm(prompt, retry_count + 1)
        
        return f"Unable to generate answer due to API error. Please try again."