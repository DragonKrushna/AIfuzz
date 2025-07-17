#!/bin/bash

# AiDirFuzz Installation Script
# This script will install AiDirFuzz on any Linux-based system

echo "============================================="
echo "    AiDirFuzz Installation Script"
echo "============================================="
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root. Please run as a regular user."
   exit 1
fi

# Check if Python 3.7+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
python_major=$(echo "$python_version" | cut -d'.' -f1)
python_minor=$(echo "$python_version" | cut -d'.' -f2)

# Check if Python version is 3.7 or higher
if [[ $python_major -eq 3 && $python_minor -ge 7 ]] || [[ $python_major -gt 3 ]]; then
    echo "✓ Python $python_version detected (compatible)"
else
    echo "Error: Python $python_version is installed, but Python 3.7 or higher is required."
    exit 1
fi



# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✓ pip3 detected"

# Create installation directory
INSTALL_DIR="$HOME/.aifuzz"
mkdir -p "$INSTALL_DIR"

# Copy main script
echo "Installing AiDirFuzz..."
cp aifuzz.py "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/aifuzz.py"

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user -r requirements_aifuzz.txt

# Install emergentintegrations if not already installed
echo "Installing AI integration package..."
pip3 install --user emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

# Create symlink for global access
SYMLINK_DIR="$HOME/.local/bin"
mkdir -p "$SYMLINK_DIR"

# Create wrapper script
cat > "$SYMLINK_DIR/aifuzz" << 'EOF'
#!/bin/bash
python3 "$HOME/.aifuzz/aifuzz.py" "$@"
EOF

chmod +x "$SYMLINK_DIR/aifuzz"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "" >> "$HOME/.bashrc"
    echo "# AiDirFuzz" >> "$HOME/.bashrc"
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.bashrc"
    echo "Added $HOME/.local/bin to PATH in .bashrc"
fi

# Create desktop shortcut (optional)
DESKTOP_DIR="$HOME/Desktop"
if [ -d "$DESKTOP_DIR" ]; then
    cat > "$DESKTOP_DIR/AiDirFuzz.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=AiDirFuzz
Comment=AI-Powered Directory Busting and Fuzzing Tool
Exec=$HOME/.local/bin/aifuzz
Icon=utilities-terminal
Terminal=true
Categories=Development;Network;Security;
EOF
    chmod +x "$DESKTOP_DIR/AiDirFuzz.desktop"
    echo "✓ Desktop shortcut created"
fi

echo ""
echo "============================================="
echo "    Installation Complete!"
echo "============================================="
echo ""
echo "AiDirFuzz has been installed successfully!"
echo ""
echo "Usage:"
echo "  aifuzz -u https://example.com -m dir -c 100"
echo "  aifuzz -u https://api.example.com -m api -c 50"
echo "  aifuzz -u https://example.com -m param -w custom_params.txt"
echo "  aifuzz -u https://example.com -m hybrid -c 200"
echo "  aifuzz --config  # Update configuration"
echo ""
echo "First run will ask for your Gemini API key."
echo "You can get one from: https://makersuite.google.com/app/apikey"
echo ""
echo "For help: aifuzz --help"
echo ""
echo "Note: If 'aifuzz' command is not found, either:"
echo "  1. Restart your terminal, or"
echo "  2. Run: source ~/.bashrc"
echo "  3. Or run directly: python3 $INSTALL_DIR/aifuzz.py"
echo ""
echo "Happy hacking! 🚀"