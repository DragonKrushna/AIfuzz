#!/bin/bash

# Complete Fresh Installation of AiFuzz v1.2.0

echo "🧹 Cleaning up any existing installations..."

# Remove any existing aifuzz installations
sudo rm -f /usr/local/bin/aifuzz /usr/bin/aifuzz /bin/aifuzz
rm -rf ~/.local/bin/aifuzz 2>/dev/null || true

# Clean Python cache
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "📦 Installing fresh AiFuzz v1.2.0..."

# Install Python dependencies
pip install -r requirements_aifuzz.txt

# Make executable
chmod +x aifuzz.py

# Create system-wide link (optional)
echo "🔗 Creating system-wide installation..."
sudo ln -sf $(pwd)/aifuzz.py /usr/local/bin/aifuzz

echo "✅ Installation complete!"
echo ""
echo "🧪 Testing installation..."
echo "Version: $(python aifuzz.py --version)"
if command -v aifuzz &> /dev/null; then
    echo "System command: $(aifuzz --version)"
else
    echo "Note: Use 'python aifuzz.py' or add to PATH"
fi
echo ""
echo "🎯 Usage:"
echo "  python aifuzz.py --help       # Local usage"
echo "  aifuzz --help                 # System-wide usage (if linked)"
echo "  aifuzz --wizard               # Interactive mode"
echo "  aifuzz --config               # Configure API keys"
echo ""
echo "📁 Files:"
echo "  Tool: $(pwd)/aifuzz.py"
echo "  Config: ~/.aifuzz/config.json"
echo "  Results: $(pwd)/aifuzz_results/"