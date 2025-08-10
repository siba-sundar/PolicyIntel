# app/utils/text_utils.py
import re
from typing import List

def words_to_numbers(text: str) -> str:
    """Convert written numbers to digits for better matching"""
    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
        'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000', 'million': '1000000'
    }
    
    for word, num in word_to_num.items():
        text = re.sub(r'\b' + word + r'\b', num, text, flags=re.IGNORECASE)
    
    return text

def extract_key_terms(question: str) -> List[str]:
    """Extract key terms from a question for better search"""
    important_terms = {
        'coverage', 'limit', 'deductible', 'premium', 'claim', 'benefit', 'exclusion',
        'copay', 'coinsurance', 'maximum', 'minimum', 'annual', 'lifetime', 'policy',
        'insured', 'covered', 'eligible', 'amount', 'percentage', 'network', 'provider',
        'emergency', 'prescription', 'medical', 'dental', 'vision', 'mental', 'health',
        'hospital', 'outpatient', 'inpatient', 'surgery', 'diagnostic', 'preventive',
        'waiting', 'period', 'authorization', 'existing', 'condition'
    }
    
    question_converted = words_to_numbers(question.lower())
    
    question_words = set(re.findall(r'\b\w+\b', question_converted))
    key_terms = list(important_terms.intersection(question_words))
    
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', question_converted)
    dollar_amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?', question_converted)
    key_terms.extend(numbers + dollar_amounts)
    
    phrases = re.findall(r'\b(?:out of pocket|prior authorization|pre existing|waiting period|per year|per day)\b', question_converted)
    key_terms.extend(phrases)
    
    return key_terms

def is_yes_no_question(question: str) -> bool:
    """Check if the question is a yes/no question"""
    question_lower = question.lower().strip()
    return question_lower.startswith(('is ', 'are ', 'do ', 'does ', 'did ', 'will ', 'would ', 'can ', 'could '))