#!/bin/bash

# Start script for South Indian Multilingual Document QA Chatbot

echo "🌏 Starting South Indian Multilingual Document QA Chatbot..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Set environment variable to suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "💡 Create a .env file with your GROQ_API_KEY to enable the chatbot"
    echo ""
fi

# Start Streamlit
echo "🚀 Launching application..."
echo "📝 Open your browser at: http://localhost:8501"
echo ""
echo "📄 Test document available: test_multilingual_doc.docx"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run main.py
