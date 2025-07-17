#!/bin/bash

# AiDirFuzz Demo Script
# This script demonstrates the capabilities of AiDirFuzz

echo "============================================="
echo "    AiDirFuzz Demo Script"
echo "============================================="
echo ""

# Check if tool is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    exit 1
fi

echo "Running AiDirFuzz demonstration..."
echo ""

# Demo 1: Basic directory scan without AI
echo "Demo 1: Basic directory scan (no AI):"
python3 aifuzz.py -u https://httpbin.org -m dir -c 5 -t 5 --no-ai -w example_wordlist.txt | head -20
echo ""

# Demo 2: Parameter fuzzing mode
echo "Demo 2: Parameter fuzzing mode:"
python3 aifuzz.py -u https://httpbin.org/get -m param -c 3 -t 5 --no-ai -w example_wordlist.txt | head -20
echo ""

# Demo 3: API discovery mode
echo "Demo 3: API discovery mode:"
python3 aifuzz.py -u https://httpbin.org -m api -c 3 -t 5 --no-ai -w example_wordlist.txt | head -20
echo ""

echo "============================================="
echo "Demo completed! Check the results above."
echo ""
echo "For full functionality with AI analysis:"
echo "1. Configure your Gemini API key: ./aifuzz.py --config"
echo "2. Run with AI: ./aifuzz.py -u https://example.com -m hybrid"
echo "============================================="