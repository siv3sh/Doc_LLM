"""
LLM Handler for Groq API integration with multilingual support.
Handles communication with Groq API and multilingual prompt construction.
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from utils import get_language_name

class GroqHandler:
    """
    Handler for Groq API integration with multilingual document QA capabilities.
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq Handler.
        
        Args:
            api_key: Groq API key (default: from environment)
            model: Groq model to use (default: mixtral-8x7b-32768)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model = model or os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Please set GROQ_API_KEY environment variable or create a .env file. "
                "Get your API key from: https://console.groq.com/keys"
            )
    
    def _construct_multilingual_prompt(self, query: str, context_chunks: List[Dict[str, Any]], language: str) -> str:
        """
        Construct a multilingual prompt for the LLM.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            language: Document language code
            
        Returns:
            str: Formatted prompt
        """
        language_name = get_language_name(language)
        
        # System prompt
        system_prompt = f"""You are a multilingual assistant specialized in South Indian languages. 
Your task is to answer questions based on the provided document context.

IMPORTANT INSTRUCTIONS:
1. Always respond in the same language as the document ({language_name})
2. Use ALL available information from the provided context
3. Extract and provide comprehensive details from the document
4. If the context contains specific names, numbers, dates, or technical details, include them in your answer
5. Be thorough and detailed in your responses
6. If multiple pieces of information are relevant, combine them for a complete answer
7. Maintain the cultural context appropriate for {language_name}

Language: {language_name} ({language})"""

        # Format context
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            context_text += f"Context {i}:\n{chunk['text']}\n\n"
        
        # User prompt
        user_prompt = f"""Context from document:
{context_text.strip()}

Question: {query}

Please provide a comprehensive answer based on ALL the provided context. Extract and include any specific details, names, numbers, dates, or technical information that is relevant to the question. If the context contains multiple relevant pieces of information, combine them for a complete answer. Respond in {language_name}."""

        return system_prompt, user_prompt
    
    def _make_api_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Make API request to Groq.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Optional[str]: Generated response or None if error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for consistent responses
            "max_tokens": 2000,  # Increased for more comprehensive responses
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], language: str) -> str:
        """
        Generate an answer using Groq API.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            language: Document language code
            
        Returns:
            str: Generated answer
        """
        try:
            if not context_chunks:
                return self._get_no_context_response(language)
            
            # Construct prompt
            system_prompt, user_prompt = self._construct_multilingual_prompt(
                query, context_chunks, language
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Make API request
            response = self._make_api_request(messages)
            
            if response:
                return response.strip()
            else:
                return self._get_error_response(language)
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return self._get_error_response(language)
    
    def _get_no_context_response(self, language: str) -> str:
        """Get response when no context is available."""
        responses = {
            'ml': "ക്ഷമിക്കണം, ഈ ചോദ്യത്തിന് ഉത്തരം നൽകാൻ ആവശ്യമായ വിവരങ്ങൾ രേഖയിൽ കണ്ടെത്താൻ കഴിഞ്ഞില്ല.",
            'ta': "மன்னிக்கவும், இந்த கேள்விக்கு பதில் அளிக்க தேவையான தகவல்களை ஆவணத்தில் காண முடியவில்லை.",
            'te': "క్షమించండి, ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి అవసరమైన సమాచారం పత్రంలో కనుగొనబడలేదు.",
            'kn': "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಅಗತ್ಯವಾದ ಮಾಹಿತಿಯನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            'tcy': "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರ ನೀಡಲು ಅಗತ್ಯವಾದ ಮಾಹಿತಿಯನ್ನು ದಾಖಲೆಯಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
            'en': "Sorry, I couldn't find sufficient information in the document to answer this question."
        }
        return responses.get(language, responses['en'])
    
    def _get_error_response(self, language: str) -> str:
        """Get response when there's an error."""
        responses = {
            'ml': "ക്ഷമിക്കണം, ഉത്തരം സൃഷ്ടിക്കുന്നതിൽ ഒരു പിശക് സംഭവിച്ചു. ദയവായി വീണ്ടും ശ്രമിക്കുക.",
            'ta': "மன்னிக்கவும், பதிலை உருவாக்குவதில் பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
            'te': "క్షమించండి, సమాధానాన్ని సృష్టించడంలో లోపం సంభవించింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
            'kn': "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ರಚಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'tcy': "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ರಚಿಸುವಲ್ಲಿ ದೋಷ ಸಂಭವಿಸಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
            'en': "Sorry, there was an error generating the response. Please try again."
        }
        return responses.get(language, responses['en'])
    
    def test_connection(self) -> bool:
        """
        Test the connection to Groq API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, please respond with 'Connection successful'."}
            ]
            
            response = self._make_api_request(test_messages)
            return response is not None
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Groq models.
        
        Returns:
            List[str]: List of available model names
        """
        return [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.1-405b-preview",
            "mixtral-8x7b-32768",
            "llama3-70b-8192", 
            "llama3-8b-8192",
            "gemma-7b-it"
        ]
    
    def set_model(self, model: str) -> bool:
        """
        Set the Groq model to use.
        
        Args:
            model: Model name
            
        Returns:
            bool: True if model is valid, False otherwise
        """
        available_models = self.get_available_models()
        if model in available_models:
            self.model = model
            return True
        else:
            print(f"Invalid model: {model}. Available models: {available_models}")
            return False
