# AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool

## 🚀 Overview

AiDirFuzz is a revolutionary penetration testing tool that combines traditional directory busting and fuzzing techniques with the power of AI. It's designed to be the most comprehensive, efficient, and intelligent tool in its category.

## ✨ Features

### 🧠 AI-Powered Analysis
- **Smart Target Analysis**: AI analyzes target URLs to suggest optimal scanning strategies
- **Intelligent Result Filtering**: AI filters false positives and identifies interesting findings
- **Vulnerability Pattern Recognition**: AI detects common security vulnerabilities and misconfigurations
- **Custom Wordlist Generation**: AI generates target-specific wordlists based on analysis
- **Adaptive Scanning**: AI adjusts scanning strategies based on target responses

### 🔧 Multiple Scanning Modes
- **Directory Discovery Mode** (`-m dir`): Traditional directory and file discovery
- **Parameter Fuzzing Mode** (`-m param`): HTTP parameter fuzzing for both GET and POST
- **API Endpoint Discovery Mode** (`-m api`): Specialized API endpoint discovery
- **Hybrid Mode** (`-m hybrid`): Combines all scanning approaches

### 🌐 GitHub Integration
- **Remote Wordlists**: Uses wordlists from GitHub repositories without downloading
- **Always Up-to-Date**: Fetches latest wordlists from SecLists and other repositories
- **No Local Storage**: Saves disk space by streaming wordlists directly

### ⚡ High Performance
- **Async HTTP Engine**: Built on aiohttp for maximum performance
- **Configurable Concurrency**: Adjustable concurrent requests (1-1000+)
- **Smart Rate Limiting**: Adaptive rate limiting to prevent server overload
- **Resume Capability**: Resume interrupted scans

### 🎨 Beautiful Interface
- **Rich CLI Interface**: Beautiful terminal interface with progress bars
- **Real-time Updates**: Live progress tracking and result display
- **Multiple Output Formats**: JSON, CSV, and TXT output formats
- **Detailed Reporting**: Comprehensive scan reports with AI insights

### 🔒 Security Features
- **SSL/TLS Support**: Full SSL/TLS certificate verification
- **Proxy Support**: HTTP/HTTPS proxy support for anonymity
- **Custom Headers**: Custom HTTP headers and cookies
- **Multiple HTTP Methods**: Support for GET, POST, PUT, DELETE, PATCH, OPTIONS

## 📋 Installation

### Prerequisites
- Python 3.7 or higher
- pip3 (Python package manager)
- Linux-based operating system

### Quick Installation
```bash
# Download and extract the tool
unzip aifuzz.zip
cd aifuzz

# Make install script executable
chmod +x install.sh

# Run installation
./install.sh
```

### Manual Installation
```bash
# Install Python dependencies
pip3 install --user -r requirements_aifuzz.txt

# Install AI integration package
pip3 install --user emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/

# Make script executable
chmod +x aifuzz.py

# Create symlink (optional)
ln -s $(pwd)/aifuzz.py ~/.local/bin/aifuzz
```

## 🚀 Quick Start

### First Time Setup
```bash
# Only if the next step doesn't work due to aiohttp related error
pip3 install --user aiohttp rich
# First run will prompt for Gemini API key
aifuzz --config
```

### Basic Usage
```bash
# Directory discovery scan
aifuzz -u https://example.com -m dir -c 100

# Parameter fuzzing
aifuzz -u https://example.com -m param -c 50

# API endpoint discovery
aifuzz -u https://api.example.com -m api -c 75

# Hybrid scan (all modes)
aifuzz -u https://example.com -m hybrid -c 200
```

## 📖 Usage Guide

### Command Line Options

```
usage: aifuzz [-h] [-u URL] [-m {dir,param,api,hybrid}] [-c CONCURRENT] 
              [-t TIMEOUT] [-d DELAY] [-w WORDLIST] [-e EXTENSIONS [EXTENSIONS ...]]
              [-s STATUS_CODES [STATUS_CODES ...]] [-o OUTPUT] [-f {json,csv,txt}]
              [--headers HEADERS [HEADERS ...]] [--cookies COOKIES [COOKIES ...]]
              [--proxy PROXY] [--no-ssl-verify] [--no-ai] [--config] 
              [--user-agent USER_AGENT] [--version]

AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool

options:
  -h, --help            show this help message and exit
  -u URL, --url URL     Target URL
  -m {dir,param,api,hybrid}, --mode {dir,param,api,hybrid}
                        Scan mode (default: dir)
  -c CONCURRENT, --concurrent CONCURRENT
                        Concurrent requests (default: 50)
  -t TIMEOUT, --timeout TIMEOUT
                        Request timeout in seconds (default: 10)
  -d DELAY, --delay DELAY
                        Delay between requests in seconds (default: 0.0)
  -w WORDLIST, --wordlist WORDLIST
                        Custom wordlist file
  -e EXTENSIONS [EXTENSIONS ...], --extensions EXTENSIONS [EXTENSIONS ...]
                        File extensions to test (e.g., php js html)
  -s STATUS_CODES [STATUS_CODES ...], --status-codes STATUS_CODES [STATUS_CODES ...]
                        Status codes to show (default: 200 201 204 301 302 307 308 403 405 500)
  -o OUTPUT, --output OUTPUT
                        Output file path
  -f {json,csv,txt}, --format {json,csv,txt}
                        Output format (default: json)
  --headers HEADERS [HEADERS ...]
                        Custom headers (format: 'Header:Value')
  --cookies COOKIES [COOKIES ...]
                        Custom cookies (format: 'name=value')
  --proxy PROXY         Proxy URL (http://proxy:port)
  --no-ssl-verify       Disable SSL verification
  --no-ai               Disable AI analysis
  --config              Update configuration
  --user-agent USER_AGENT
                        Custom User-Agent
  --version             show program's version number and exit
```

### Scanning Modes

#### 1. Directory Discovery Mode (`-m dir`)
Discovers hidden directories and files on web servers.

```bash
# Basic directory scan
aifuzz -u https://example.com -m dir

# With file extensions
aifuzz -u https://example.com -m dir -e php js html txt

# With custom wordlist
aifuzz -u https://example.com -m dir -w /path/to/wordlist.txt

# High concurrency scan
aifuzz -u https://example.com -m dir -c 200
```

#### 2. Parameter Fuzzing Mode (`-m param`)
Fuzzes HTTP parameters to find hidden functionality.

```bash
# Basic parameter fuzzing
aifuzz -u https://example.com -m param

# With custom parameter wordlist
aifuzz -u https://example.com -m param -w params.txt

# With rate limiting
aifuzz -u https://example.com -m param -c 30 -d 0.1
```

#### 3. API Endpoint Discovery Mode (`-m api`)
Specialized for discovering API endpoints.

```bash
# Basic API discovery
aifuzz -u https://api.example.com -m api

# With custom headers
aifuzz -u https://api.example.com -m api --headers "Authorization:Bearer token"

# Multiple HTTP methods
aifuzz -u https://api.example.com -m api -c 100
```

#### 4. Hybrid Mode (`-m hybrid`)
Combines all scanning approaches for comprehensive testing.

```bash
# Full hybrid scan
aifuzz -u https://example.com -m hybrid -c 150

# Hybrid with output
aifuzz -u https://example.com -m hybrid -o results.json
```

### Advanced Usage

#### Custom Headers and Cookies
```bash
# With authentication
aifuzz -u https://example.com -m dir \
  --headers "Authorization:Bearer token" "X-Custom:value" \
  --cookies "session=abc123" "csrf=xyz789"

# With proxy
aifuzz -u https://example.com -m dir --proxy http://proxy:8080
```

#### Output Formats
```bash
# JSON output (default)
aifuzz -u https://example.com -m dir -o results.json

# CSV output
aifuzz -u https://example.com -m dir -o results.csv -f csv

# Text output
aifuzz -u https://example.com -m dir -o results.txt -f txt
```

#### Status Code Filtering
```bash
# Show only specific status codes
aifuzz -u https://example.com -m dir -s 200 403 500

# Show all status codes
aifuzz -u https://example.com -m dir -s 200 301 302 400 401 403 404 500 502 503
```

## 🤖 AI Integration

### Gemini API Setup
1. Get API key from: https://makersuite.google.com/app/apikey
2. Configure during first run or update with `--config`

### AI Features

#### Target Analysis
AI analyzes the target URL to:
- Detect technology stack
- Suggest relevant directories/files
- Recommend optimal scanning parameters
- Identify potential attack vectors

#### Result Analysis
AI analyzes each HTTP response to:
- Calculate vulnerability scores
- Identify interesting findings
- Filter false positives
- Detect security misconfigurations

#### Custom Wordlist Generation
AI generates target-specific wordlists based on:
- Domain name analysis
- Technology stack detection
- Common patterns for the target type
- Security testing best practices

## 📊 Output Formats

### JSON Output
```json
[
  {
    "url": "https://example.com/admin",
    "method": "GET",
    "status_code": 200,
    "content_length": 1234,
    "content_type": "text/html",
    "response_time": 123.45,
    "redirect_url": null,
    "ai_analysis": "Potential admin panel found. High security risk.",
    "vulnerability_score": 8.5,
    "timestamp": "2024-01-01T12:00:00"
  }
]
```

### CSV Output
```csv
url,method,status_code,content_length,content_type,response_time,vulnerability_score,ai_analysis,timestamp
https://example.com/admin,GET,200,1234,text/html,123.45,8.5,"Potential admin panel found",2024-01-01T12:00:00
```

### Text Output
```
200 GET https://example.com/admin
403 GET https://example.com/config
500 POST https://example.com/api/users
```

## 🔧 Configuration

### Configuration File
Located at `~/.aifuzz/config.json`:

```json
{
  "gemini_api_key": "your-api-key-here",
  "gemini_model": "gemini-2.0-flash"
}
```

### Updating Configuration
```bash
# Update API key and model
aifuzz --config
```

### Available Gemini Models
- `gemini-2.5-flash-preview-04-17` (Latest)
- `gemini-2.5-pro-preview-05-06` (Most capable)
- `gemini-2.0-flash` (Recommended)
- `gemini-2.0-flash-lite` (Faster)
- `gemini-1.5-flash` (Stable)
- `gemini-1.5-pro` (Advanced)

## 🎯 Examples

### Web Application Testing
```bash
# Full web app assessment
aifuzz -u https://webapp.example.com -m hybrid -c 100 -o webapp_results.json

# Focus on admin areas
aifuzz -u https://webapp.example.com -m dir -w admin_wordlist.txt -c 50

# Parameter fuzzing for login
aifuzz -u https://webapp.example.com/login -m param -c 30
```

### API Testing
```bash
# API discovery
aifuzz -u https://api.example.com -m api -c 75 -o api_results.json

# With authentication
aifuzz -u https://api.example.com -m api \
  --headers "Authorization:Bearer token" \
  --cookies "session=abc123"
```

### Stealth Scanning
```bash
# Slow and steady scan
aifuzz -u https://example.com -m dir -c 10 -d 0.5 --proxy http://proxy:8080

# Custom user agent
aifuzz -u https://example.com -m dir --user-agent "Mozilla/5.0 (compatible; bot)"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Installation Issues
```bash
# Update pip
python3 -m pip install --upgrade pip

# Install with user flag
pip3 install --user -r requirements_aifuzz.txt

# Check Python version
python3 --version
```

#### 2. AI Integration Issues
```bash
# Check API key
aifuzz --config

# Test without AI
aifuzz -u https://example.com -m dir --no-ai

# Install AI package manually
pip3 install emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/
```

#### 3. Network Issues
```bash
# Use proxy
aifuzz -u https://example.com -m dir --proxy http://proxy:8080

# Disable SSL verification
aifuzz -u https://example.com -m dir --no-ssl-verify

# Reduce concurrency
aifuzz -u https://example.com -m dir -c 10
```

#### 4. Performance Issues
```bash
# Increase timeout
aifuzz -u https://example.com -m dir -t 30

# Add delay between requests
aifuzz -u https://example.com -m dir -d 0.1

# Reduce concurrent requests
aifuzz -u https://example.com -m dir -c 25
```

## 🛡️ Legal Disclaimer

**IMPORTANT: Use this tool responsibly and only on systems you own or have explicit permission to test.**

This tool is designed for:
- Security professionals conducting authorized penetration tests
- Bug bounty hunters testing in-scope applications
- Developers testing their own applications
- Security researchers in controlled environments

**DO NOT use this tool for:**
- Unauthorized testing of systems you don't own
- Illegal activities or malicious purposes
- Violating terms of service or applicable laws
- Causing harm to systems or services

The authors are not responsible for any misuse of this tool. Always ensure you have proper authorization before testing any system.

## 📝 Changelog

### Version 1.0.0
- Initial release
- AI-powered analysis with Gemini integration
- Multiple scanning modes (dir, param, api, hybrid)
- GitHub wordlist integration
- High-performance async HTTP engine
- Beautiful CLI interface
- Multiple output formats
- Comprehensive documentation

## 🤝 Contributing

This tool is designed to be the most comprehensive directory busting and fuzzing tool available. If you have suggestions for improvements or new features, please consider:

1. Testing the tool thoroughly
2. Reporting bugs and issues
3. Suggesting new features
4. Contributing wordlists
5. Improving documentation

## 📄 License

This tool is released under the MIT License. See the LICENSE file for more details.

## 🙏 Acknowledgments

- SecLists project for comprehensive wordlists
- Google Gemini for AI capabilities
- The penetration testing community for inspiration
- Rich library for beautiful terminal interface
- aiohttp for high-performance HTTP client

## 📞 Support

For support, questions, or feature requests, please check the documentation first. This tool is designed to be self-contained and comprehensive.

---

**Happy Ethical Hacking! 🚀**
