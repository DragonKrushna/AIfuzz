#!/bin/bash

# AiFuzz Installation Script
# This script installs the improved AiFuzz tool

set -e

echo "Installing AiFuzz - Improved Version..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements_aifuzz.txt

# Make the script executable
chmod +x aifuzz_improved.py

# Create symlink for easy access
if [ ! -L "/usr/local/bin/aifuzz" ]; then
    sudo ln -sf "$(pwd)/aifuzz_improved.py" /usr/local/bin/aifuzz
fi

echo "Installation complete!"
echo ""
echo "Usage:"
echo "  aifuzz --help                    # Show help"
echo "  aifuzz --wizard                  # Interactive wizard mode"
echo "  aifuzz --config                  # Configure API keys"
echo "  aifuzz -u https://example.com    # Quick scan"
echo ""
echo "First-time setup:"
echo "  1. Run 'aifuzz --config' to set up your Gemini API key"
echo "  2. Run 'aifuzz --wizard' for interactive configuration"
echo ""