# AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool
## World's First AI-Powered Penetration Testing Tool

### 🚀 Quick Start
```bash
# Extract the tool
unzip aifuzz.zip

# Run installation (Linux only)
chmod +x install.sh
./install.sh

# Configure your Gemini API key
aifuzz --config

# Start scanning
aifuzz -u https://example.com -m dir -c 100
```

### 🎯 Features
- **AI-Powered Analysis**: Smart target analysis, vulnerability scoring, and false positive filtering
- **Multiple Modes**: Directory discovery, parameter fuzzing, API endpoint discovery, and hybrid mode
- **GitHub Integration**: Uses wordlists from GitHub repositories without downloading
- **High Performance**: Async HTTP engine with configurable concurrency (1-1000+ requests)
- **Beautiful Interface**: Rich CLI with progress bars and detailed reporting

### 📁 Files Included
- `aifuzz.py` - Main tool executable
- `requirements_aifuzz.txt` - Python dependencies
- `install.sh` - Linux installation script
- `README.md` - Comprehensive documentation
- `LICENSE` - MIT License
- `example_wordlist.txt` - Sample wordlist for testing
- `demo.sh` - Demonstration script

### 🔧 Quick Examples
```bash
# Directory discovery
aifuzz -u https://example.com -m dir -c 100

# Parameter fuzzing
aifuzz -u https://example.com -m param -c 50

# API discovery
aifuzz -u https://api.example.com -m api -c 75

# Hybrid mode (all methods)
aifuzz -u https://example.com -m hybrid -c 200

# With custom wordlist
aifuzz -u https://example.com -m dir -w custom_wordlist.txt

# Output to file
aifuzz -u https://example.com -m dir -o results.json
```

### 🤖 AI Features
- **Smart Target Analysis**: AI analyzes target URLs for optimal scanning
- **Vulnerability Scoring**: AI scores findings based on security implications
- **Custom Wordlist Generation**: AI creates target-specific wordlists
- **False Positive Filtering**: AI reduces noise and identifies real issues
- **Pattern Recognition**: AI detects common vulnerability patterns

### 🛡️ Legal Notice
Use responsibly and only on systems you own or have explicit permission to test. This tool is for authorized security testing only.

### 📞 Support
For help: `aifuzz --help`
For configuration: `aifuzz --config`
For demonstration: `./demo.sh`

**Happy Ethical Hacking! 🚀**