"""
Utility functions for multilingual document processing.
Handles text extraction, language detection, and text cleaning.
"""

import os
import tempfile
import re
import pdfplumber
import docx2txt
from typing import Optional, Dict, List, Tuple

# Language code mapping for South Indian languages
LANGUAGE_MAPPING = {
    'ml': 'Malayalam',
    'ta': 'Tamil', 
    'te': 'Telugu',
    'kn': 'Kannada',
    'hi': 'Hindi',
    'tcy': 'Tulu',  # Tulu language code
    'en': 'English'
}

# Unicode ranges for South Indian scripts
UNICODE_RANGES = {
    'hi': (0x0900, 0x097F),  # Devanagari (Hindi)
    'ml': (0x0D00, 0x0D7F),  # Malayalam
    'ta': (0x0B80, 0x0BFF),  # Tamil
    'te': (0x0C00, 0x0C7F),  # Telugu
    'kn': (0x0C80, 0x0CFF),  # Kannada
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
        elif file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif']:
            return extract_from_image(file)
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
        
        # If still empty, fallback to OCR for image-based PDFs
        if not text_content:
            try:
                ocr_text = ocr_pdf(file)
                if ocr_text and ocr_text.strip():
                    return ocr_text
            except Exception as ocr_err:
                print(f"OCR fallback failed: {ocr_err}")
        
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
            if text_content:
                return '\n'.join(text_content)
            # If PyPDF2 also failed to get text, try OCR
            ocr_text = ocr_pdf(file)
            return ocr_text
        except Exception as e2:
            print(f"Alternative PDF extraction also failed: {e2}")
            raise Exception(f"Could not extract text from PDF: {e}")

def ocr_pdf(file) -> str:
    """OCR fallback: render PDF pages to images and extract text via Tesseract.
    Requires poppler (for pdf2image) and Tesseract installed on system.
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        from PIL import Image
    except Exception as import_err:
        raise Exception(
            "OCR dependencies missing. Please install system packages and pip deps: "
            "poppler (system), tesseract-ocr (system), and pip install pdf2image pytesseract pillow"
        ) from import_err

    # Read file bytes for conversion
    try:
        file.seek(0)
        pdf_bytes = file.read()
    except Exception:
        # Some uploaders require getbuffer
        try:
            file.seek(0)
            pdf_bytes = file.getbuffer().tobytes()  # type: ignore[attr-defined]
        except Exception as e:
            raise Exception(f"Failed to read PDF bytes for OCR: {e}")

    # Convert PDF to images
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300)
    except Exception as conv_err:
        raise Exception(
            "Failed to render PDF pages for OCR. Ensure poppler is installed and available in PATH."
        ) from conv_err

    # Configure Tesseract language set; include Indic scripts
    tesseract_langs = 'eng+hin+mal+tam+tel+kan'
    ocr_texts: List[str] = []
    for idx, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(img, lang=tesseract_langs)
            if text and text.strip():
                ocr_texts.append(text.strip())
        except Exception as ocr_page_err:
            print(f"Multilingual OCR failed on page {idx+1}, trying English: {ocr_page_err}")
            # Fallback to English-only OCR
            try:
                text = pytesseract.image_to_string(img, lang='eng')
                if text and text.strip():
                    ocr_texts.append(text.strip())
            except Exception as eng_err:
                print(f"English OCR also failed on page {idx+1}: {eng_err}")
                # Try basic OCR without language specification
                try:
                    text = pytesseract.image_to_string(img)
                    if text and text.strip():
                        ocr_texts.append(text.strip())
                except Exception:
                    continue  # Skip this page if all methods fail
    return '\n\n'.join(ocr_texts) if ocr_texts else ""

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
                            # Only count if we got a valid language
                            if lang in LANGUAGE_MAPPING:
                                counts[lang] = counts.get(lang, 0) + 1
            except Exception as e:
                print(f"Language distribution PDF error: {e}")
        elif extension in ('docx', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif'):
            # Reuse existing extractors for full text detection
            file.seek(0)
            text = extract_text(file)
            lang = detect_language(text or '')
            # Only count if we got a valid language
            if lang in LANGUAGE_MAPPING:
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
    
    # Save temporary file in a writable temp directory (works on Streamlit Cloud)
    tmp_dir = tempfile.gettempdir()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=tmp_dir) as tmp:
        temp_path = tmp.name
        tmp.write(file.getbuffer())
    
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

def extract_from_image(file) -> str:
    """Extract text from image files using OCR (Tesseract).
    Supports multilingual text extraction including Indic scripts.
    
    Args:
        file: Streamlit uploaded file object (image)
        
    Returns:
        str: Extracted text from the image
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as import_err:
        raise Exception(
            "OCR dependencies missing. Please install: pip install Pillow pytesseract\n"
            "System requirement: Install tesseract-ocr\n"
            "  - macOS: brew install tesseract tesseract-lang\n"
            "  - Ubuntu/Debian: apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin "
            "tesseract-ocr-mal tesseract-ocr-tam tesseract-ocr-tel tesseract-ocr-kan\n"
            "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
        ) from import_err
    
    try:
        # Reset file pointer
        file.seek(0)
        
        # Open image using PIL
        image = Image.open(file)
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Configure Tesseract with support for multiple Indian languages
        # Languages: English, Hindi, Malayalam, Tamil, Telugu, Kannada
        tesseract_langs = 'eng+hin+mal+tam+tel+kan'
        
        # Perform OCR with enhanced configuration for better accuracy
        custom_config = r'--oem 3 --psm 6'  # OEM 3 = Default, PSM 6 = Assume uniform block of text
        
        try:
            text = pytesseract.image_to_string(
                image,
                lang=tesseract_langs,
                config=custom_config
            )
        except Exception as tesseract_err:
            # Fallback: try with English only if multilingual fails
            print(f"Multilingual OCR failed, trying English only: {tesseract_err}")
            try:
                text = pytesseract.image_to_string(image, lang='eng', config=custom_config)
            except Exception as eng_err:
                # Last resort: try without any config
                print(f"English OCR failed, trying basic extraction: {eng_err}")
                try:
                    text = pytesseract.image_to_string(image)
                except Exception:
                    text = ""  # Return empty rather than fail
        
        # Return whatever text was extracted, even if minimal
        if not text:
            text = ""  # Return empty string instead of raising error
        
        return text.strip()
        
    except Exception as e:
        if "extract_from_image" in str(e) or "OCR dependencies" in str(e):
            raise
        raise Exception(f"Error processing image: {str(e)}")

def detect_language(text: str) -> str:
    """
    Detect language of the input text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        str: Language code (e.g., 'ml', 'ta', 'te', 'kn', 'tcy', 'en')
    """
    try:
        # Clean text for better detection
        cleaned_text = clean_text_for_detection(text)
        
        # Process any length of text - no minimum requirement
        if len(cleaned_text.strip()) == 0:
            return 'en'  # Default to English if text is empty
        
        # First, use character-based detection for South Indian languages
        south_indian_lang = detect_south_indian_language(cleaned_text)
        
        if south_indian_lang != 'en':
            return south_indian_lang
        
        # If no South Indian language detected, try langdetect for English
        try:
            from langdetect import detect
            detected_lang = detect(cleaned_text)
            
            # Map detected language to our supported languages
            if detected_lang in LANGUAGE_MAPPING:
                return detected_lang
        except Exception:
            pass
        
        return 'en'  # Default fallback
            
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en'  # Default fallback

def detect_south_indian_language(text: str) -> str:
    """
    Enhanced detection for Indic languages using Unicode character patterns.
    Includes Hindi (Devanagari) and major South Indian scripts.
    """
    if not text or len(text.strip()) == 0:
        return 'en'
    
    # Count characters in each relevant script
    hindi_chars = re.findall(r'[\u0900-\u097F]', text)       # Devanagari (Hindi)
    malayalam_chars = re.findall(r'[\u0D00-\u0D7F]', text)
    tamil_chars = re.findall(r'[\u0B80-\u0BFF]', text)
    telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
    kannada_chars = re.findall(r'[\u0C80-\u0CFF]', text)
    
    char_counts = {
        'hi': len(hindi_chars),
        'ml': len(malayalam_chars),
        'ta': len(tamil_chars),
        'te': len(telugu_chars),
        'kn': len(kannada_chars)
    }
    
    # Calculate percentages to handle mixed-language texts
    total_chars = sum(char_counts.values())
    
    if total_chars == 0:
        return 'en'  # No South Indian script detected
    
    # If one language clearly dominates (>80% of characters)
    for lang, count in char_counts.items():
        if count > 0 and (count / total_chars) > 0.8:
            return lang
    
    # Otherwise, return the language with the highest count
    if max(char_counts.values()) > 0:
        max_lang = max(char_counts, key=char_counts.get)
        return max_lang
    
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

def validate_text_length(text: str, min_length: int = 0) -> bool:
    """
    Validate if extracted text meets minimum length requirements.
    Now accepts any text - always returns True unless completely empty.
    
    Args:
        text: Extracted text
        min_length: Minimum required length (default 0 - accepts anything)
        
    Returns:
        bool: True if text has any content, False only if completely empty
    """
    return text is not None and len(text.strip()) >= min_length
