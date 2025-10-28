"""
Utility functions for multilingual document processing.
Handles text extraction, language detection, and text cleaning.
"""

import os
import re
import pdfplumber
import docx2txt
from langdetect import detect, DetectorFactory
from typing import Optional, Dict, List, Tuple

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Language code mapping for South Indian languages
LANGUAGE_MAPPING = {
    'ml': 'Malayalam',
    'ta': 'Tamil', 
    'te': 'Telugu',
    'kn': 'Kannada',
    'tcy': 'Tulu',  # Tulu language code
    'en': 'English'
}

def extract_text(file) -> str:
    """
    Extract text from uploaded file based on file type.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        str: Extracted text content
    """
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'pdf':
            return extract_from_pdf(file)
        elif file_extension == 'docx':
            return extract_from_docx(file)
        elif file_extension == 'txt':
            return extract_from_txt(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        raise Exception(f"Error extracting text from {file.name}: {str(e)}")

def extract_from_pdf(file) -> str:
    """Extract text from PDF using pdfplumber."""
    text_content = []
    
    # Reset file pointer
    file.seek(0)
    
    try:
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_content.append(page_text.strip())
                    else:
                        # Try alternative extraction method
                        page_text = page.extract_text_simple()
                        if page_text and page_text.strip():
                            text_content.append(page_text.strip())
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    continue
        
        if not text_content:
            # Try with different settings
            file.seek(0)
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    try:
                        # Try with different extraction parameters
                        page_text = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=False,
                            x_density=7.25,
                            y_density=13
                        )
                        if page_text and page_text.strip():
                            text_content.append(page_text.strip())
                    except Exception:
                        continue
        
        return '\n'.join(text_content)
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        # Try alternative PDF processing
        try:
            import PyPDF2
            file.seek(0)
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(page_text.strip())
            return '\n'.join(text_content)
        except Exception as e2:
            print(f"Alternative PDF extraction also failed: {e2}")
            raise Exception(f"Could not extract text from PDF: {e}")

def get_language_distribution(file) -> Dict[str, int]:
    """Compute language distribution in a document. For PDFs, detects per-page languages.
    For DOCX/TXT, detects language once for the whole document.

    Args:
        file: Streamlit uploaded file object

    Returns:
        Dict[str, int]: Mapping of language code to count (pages for PDF, 1 for others)
    """
    try:
        extension = file.name.split('.')[-1].lower()
        counts: Dict[str, int] = {}
        if extension == 'pdf':
            file.seek(0)
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text() or page.extract_text_simple()
                        except Exception:
                            page_text = None
                        if page_text and page_text.strip():
                            lang = detect_language(page_text)
                            counts[lang] = counts.get(lang, 0) + 1
            except Exception as e:
                print(f"Language distribution PDF error: {e}")
        elif extension in ('docx', 'txt'):
            # Reuse existing extractors for full text detection
            text = extract_text(file)
            lang = detect_language(text or '')
            counts[lang] = counts.get(lang, 0) + 1
        else:
            # Unsupported types handled elsewhere
            pass
        return counts
    except Exception as e:
        print(f"get_language_distribution error: {e}")
        return {}

def extract_from_docx(file) -> str:
    """Extract text from DOCX using docx2txt."""
    # Reset file pointer
    file.seek(0)
    
    # Save temporary file
    temp_path = f"temp_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.getbuffer())
    
    try:
        text = docx2txt.process(temp_path)
        return text
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def extract_from_txt(file) -> str:
    """Extract text from TXT file."""
    # Reset file pointer
    file.seek(0)
    
    # Try different encodings
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            file.seek(0)
            return file.read().decode(encoding)
        except UnicodeDecodeError:
            continue
    
    raise Exception("Could not decode text file with any supported encoding")

def detect_language(text: str) -> str:
    """
    Detect language of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        str: Language code (e.g., 'ml', 'ta', 'te', 'kn', 'tcy')
    """
    try:
        # Clean text for better detection
        cleaned_text = clean_text_for_detection(text)
        
        if len(cleaned_text.strip()) < 10:
            return 'en'  # Default to English if text is too short
        
        detected_lang = detect(cleaned_text)
        
        # Map detected language to our supported languages
        if detected_lang in LANGUAGE_MAPPING:
            return detected_lang
        else:
            # For South Indian languages that might be misdetected
            return detect_south_indian_language(cleaned_text)
            
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en'  # Default fallback

def detect_south_indian_language(text: str) -> str:
    """
    Enhanced detection for South Indian languages using character patterns.
    """
    # Character ranges for South Indian scripts
    malayalam_chars = re.findall(r'[\u0D00-\u0D7F]', text)
    tamil_chars = re.findall(r'[\u0B80-\u0BFF]', text)
    telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
    kannada_chars = re.findall(r'[\u0C80-\u0CFF]', text)
    
    char_counts = {
        'ml': len(malayalam_chars),
        'ta': len(tamil_chars),
        'te': len(telugu_chars),
        'kn': len(kannada_chars)
    }
    
    # Return language with highest character count
    if max(char_counts.values()) > 0:
        return max(char_counts, key=char_counts.get)
    
    return 'en'  # Default to English

def get_language_name(lang_code: str) -> str:
    """
    Get human-readable language name from language code.
    
    Args:
        lang_code: Language code (e.g., 'ml', 'ta')
        
    Returns:
        str: Language name (e.g., 'Malayalam', 'Tamil')
    """
    return LANGUAGE_MAPPING.get(lang_code, 'Unknown')

def clean_text_for_detection(text: str) -> str:
    """
    Clean text specifically for language detection.
    """
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common English words that might interfere
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in english_words]
    
    return ' '.join(filtered_words[:100])  # Use first 100 words for detection

def clean_text(text: str) -> str:
    """
    Clean extracted text for better processing.
    
    Args:
        text: Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Page \d+$', '', text, flags=re.MULTILINE)
    
    # Remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    return '\n'.join(lines)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def validate_text_length(text: str, min_length: int = 20) -> bool:
    """
    Validate if extracted text meets minimum length requirements.
    
    Args:
        text: Extracted text
        min_length: Minimum required length
        
    Returns:
        bool: True if text is valid, False otherwise
    """
    return len(text.strip()) >= min_length
