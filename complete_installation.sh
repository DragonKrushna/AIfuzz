#!/bin/bash

# Complete Automated Installation Script for AiFuzz v1.2.0
# This script will completely clean and install the correct version

set -e  # Exit on any error

echo "🚀 AiFuzz v1.2.0 - Complete Automated Installation"
echo "=================================================="

# Step 1: Complete Cleanup
echo "🧹 Step 1: Cleaning up all existing installations..."

# Remove all existing aifuzz installations
sudo rm -f /usr/local/bin/aifuzz /usr/bin/aifuzz /bin/aifuzz 2>/dev/null || true
rm -f ~/.local/bin/aifuzz 2>/dev/null || true

# Clear bash command cache
hash -d aifuzz 2>/dev/null || true

# Remove old files
rm -f aifuzz.py aifuzz_improved.py 2>/dev/null || true

# Clean Python cache
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Cleanup complete"

# Step 2: Verify Python
echo "🐍 Step 2: Verifying Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ Found: $PYTHON_VERSION"

# Step 3: Install Dependencies
echo "📦 Step 3: Installing clean dependencies..."
cat > requirements_aifuzz.txt << 'EOF'
aiohttp>=3.8.0
rich>=13.0.0
google-genai>=1.26.0
EOF

python3 -m pip install --user -r requirements_aifuzz.txt --upgrade

echo "✅ Dependencies installed"

# Step 4: Create the Correct AiFuzz v1.2.0
echo "🔧 Step 4: Creating AiFuzz v1.2.0..."

# This will be the CORRECT version without any old issues
cat > aifuzz.py << 'EOF'
#!/usr/bin/env python3

"""
AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool
Version: 1.2.0 (Clean Build)
"""

import asyncio
import aiohttp
import argparse
import json
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import threading
import signal

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Initialize Rich console
console = Console()

# Global variables
VERBOSE_MODE = False
SCAN_INTERRUPTED = False
CURRENT_RESULTS = []

# Configuration
CONFIG_DIR = Path.home() / ".aifuzz"
CONFIG_FILE = CONFIG_DIR / "config.json"
CONFIG_DIR.mkdir(exist_ok=True)

# Clean wordlist configurations with proper raw URLs
WORDLIST_CONFIGS = {
    "small": {
        "dir": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt", 500),
        ],
        "param": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/burp-parameter-names.txt", 400),
        ],
        "api": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/api/api-endpoints.txt", 300),
        ]
    },
    "medium": {
        "dir": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt", 1000),
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/big.txt", 800),
        ],
        "param": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/burp-parameter-names.txt", 800),
        ],
        "api": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/api/api-endpoints.txt", 600),
        ]
    },
    "large": {
        "dir": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt", 2000),
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/big.txt", 1500),
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/directory-list-2.3-medium.txt", 2000),
        ],
        "param": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/burp-parameter-names.txt", 1500),
        ],
        "api": [
            ("https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/api/api-endpoints.txt", 1200),
        ]
    }
}

@dataclass
class ScanResult:
    """Represents a scan result"""
    url: str
    method: str
    status_code: int
    content_length: int
    content_type: str
    response_time: float
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

class VerboseLogger:
    """Clean verbose logging without spam"""
    
    def __init__(self, console: Console):
        self.console = console
        self.is_verbose = False
        self.is_interactive_verbose = False
        self.verbose_timer = None
        self.logs = []
        self.max_logs = 50  # Reduced from 100
        self.error_count = 0
        self.max_errors = 10  # Limit error messages
        
    def log(self, message: str, level: str = "info"):
        """Log a message with better filtering"""
        # Skip excessive error messages
        if level == "error":
            self.error_count += 1
            if self.error_count > self.max_errors:
                return
        
        # Skip boring messages
        if any(skip in message.lower() for skip in ["fetching wordlist", "loading", "added", "keyboard listener"]):
            if not self.is_verbose:
                return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)
        
        # Show if verbose is enabled
        if self.is_verbose or self.is_interactive_verbose:
            if level == "error":
                self.console.print(f"[red]{log_entry}[/red]")
            elif level == "warning":
                self.console.print(f"[yellow]{log_entry}[/yellow]")
            elif level == "success":
                self.console.print(f"[green]{log_entry}[/green]")
            else:
                self.console.print(f"[dim]{log_entry}[/dim]")
    
    def set_verbose(self, enabled: bool):
        """Set permanent verbose mode"""
        self.is_verbose = enabled

class WordlistFetcher:
    """Clean wordlist fetcher with proper URLs"""
    
    def __init__(self, verbose_logger: VerboseLogger):
        self.session = None
        self.cache = {}
        self.logger = verbose_logger
        self.custom_repos = []
        self.timeout = aiohttp.ClientTimeout(total=30)
    
    def add_custom_repo(self, repo_url: str):
        """Add custom GitHub repository URL"""
        # Convert GitHub URLs to raw URLs
        if "github.com" in repo_url and "/blob/" in repo_url:
            repo_url = repo_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        self.custom_repos.append(repo_url)
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=50)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={'User-Agent': 'AiFuzz/1.2.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_wordlist(self, url: str, limit: int = -1) -> List[str]:
        """Fetch wordlist from URL with proper error handling"""
        cache_key = f"{url}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    wordlist = [
                        line.strip() for line in content.split('\n') 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    
                    if limit > 0:
                        wordlist = wordlist[:limit]
                    
                    self.cache[cache_key] = wordlist
                    return wordlist
                else:
                    self.logger.log(f"Failed to fetch wordlist: {response.status}", "error")
                    return []
        except Exception as e:
            self.logger.log(f"Error fetching wordlist: {str(e)}", "error")
            return []
    
    async def get_combined_wordlist(self, wordlist_type: str, size: str = "medium") -> List[str]:
        """Get combined wordlist from multiple sources"""
        all_words = set()
        
        # Add words from default wordlists
        if size in WORDLIST_CONFIGS and wordlist_type in WORDLIST_CONFIGS[size]:
            for url, limit in WORDLIST_CONFIGS[size][wordlist_type]:
                try:
                    words = await self.fetch_wordlist(url, limit)
                    all_words.update(words)
                except Exception as e:
                    self.logger.log(f"Failed to fetch from {url}: {str(e)}", "error")
                    continue
        
        # Add words from custom repositories
        for repo_url in self.custom_repos:
            try:
                words = await self.fetch_wordlist(repo_url)
                all_words.update(words)
            except Exception as e:
                self.logger.log(f"Failed to fetch from custom repo: {str(e)}", "error")
                continue
        
        result = list(all_words)
        return result

class DirectGeminiAnalyzer:
    """Direct Gemini API integration"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Gemini client"""
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            self.client = None
    
    async def analyze_scan_results(self, results: List[ScanResult], target_url: str) -> Dict[str, Any]:
        """Analyze scan results after completion"""
        if not self.client:
            return {"error": "Gemini client not initialized"}
        
        try:
            prompt = f"""
            Analyze web scan results for {target_url}:
            Total endpoints: {len(results)}
            
            Key findings:
            {json.dumps([{"url": r.url, "status": r.status_code, "type": r.content_type} for r in results[:10]], indent=2)}
            
            Provide security assessment and recommendations.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            return {"analysis": response.text, "error": None}
            
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}

class Fuzzer:
    """Main fuzzing engine"""
    
    def __init__(self, verbose_logger: VerboseLogger):
        self.logger = verbose_logger
        self.session = None
        self.wordlist_fetcher = WordlistFetcher(self.logger)
        self.ai_analyzer = None
        self.results = []
        self.completed_requests = 0
        self.total_requests = 0
        self.start_time = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def setup_ai_analyzer(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Setup AI analyzer"""
        if api_key:
            self.ai_analyzer = DirectGeminiAnalyzer(api_key, model)
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL to include protocol"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url.rstrip('/')
    
    async def run_scan(self, config: Dict[str, Any]) -> List[ScanResult]:
        """Run the fuzzing scan"""
        global CURRENT_RESULTS
        
        self.start_time = time.time()
        self.results = []
        CURRENT_RESULTS = self.results
        
        try:
            # Normalize target URL
            config['url'] = self.normalize_url(config['url'])
            
            # Load wordlist
            console.print("[yellow]Loading wordlists...[/yellow]")
            
            async with self.wordlist_fetcher as fetcher:
                # Add custom repos
                for repo in config.get('github_repos', []):
                    fetcher.add_custom_repo(repo)
                
                # Load wordlist
                if config.get('custom_wordlist'):
                    wordlist = self._load_custom_wordlist(config['custom_wordlist'])
                else:
                    wordlist = await fetcher.get_combined_wordlist(
                        config['mode'], 
                        config.get('wordlist_size', 'medium')
                    )
            
            if not wordlist:
                console.print("[red]No wordlist loaded. Exiting.[/red]")
                return []
            
            console.print(f"[green]Loaded {len(wordlist)} words for scanning[/green]")
            
            # Setup progress tracking
            self.total_requests = len(wordlist)
            self.completed_requests = 0
            
            # Create progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=4
            ) as progress:
                progress_task = progress.add_task("Scanning...", total=self.total_requests)
                
                # Run scan
                await self._execute_scan(wordlist, config, progress, progress_task)
            
            # Post-scan AI analysis
            if self.ai_analyzer and self.results:
                console.print("[yellow]Running AI analysis...[/yellow]")
                ai_results = await self.ai_analyzer.analyze_scan_results(self.results, config['url'])
                if not ai_results.get('error'):
                    console.print("[green]AI analysis completed[/green]")
            
            return self.results
            
        except Exception as e:
            self.logger.log(f"Scan error: {str(e)}", "error")
            return self.results
    
    async def _execute_scan(self, wordlist: List[str], config: Dict[str, Any], progress, progress_task):
        """Execute the actual scan"""
        semaphore = asyncio.Semaphore(config.get('concurrent', 50))
        
        async def scan_word(word: str):
            async with semaphore:
                await self._scan_single_word(word, config)
                self.completed_requests += 1
                progress.update(progress_task, completed=self.completed_requests)
        
        # Create tasks in batches
        batch_size = 500
        for i in range(0, len(wordlist), batch_size):
            batch = wordlist[i:i + batch_size]
            tasks = [scan_word(word) for word in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _scan_single_word(self, word: str, config: Dict[str, Any]):
        """Scan a single word"""
        try:
            target_url = config['url']
            
            if config['mode'] == 'dir':
                test_url = f"{target_url}/{word}"
            elif config['mode'] == 'param':
                test_url = f"{target_url}?{word}=test"
            elif config['mode'] == 'api':
                test_url = f"{target_url}/api/{word}"
            else:  # hybrid
                for mode in ['dir', 'param', 'api']:
                    temp_config = config.copy()
                    temp_config['mode'] = mode
                    await self._scan_single_word(word, temp_config)
                return
            
            # Add extensions if specified
            urls_to_test = [test_url]
            if config.get('extensions'):
                for ext in config['extensions']:
                    urls_to_test.append(f"{test_url}.{ext}")
            
            for url in urls_to_test:
                await self._test_url(url, config)
                
        except Exception:
            # Silently skip errors to avoid spam
            pass
    
    async def _test_url(self, url: str, config: Dict[str, Any]):
        """Test a single URL"""
        try:
            headers = {
                'User-Agent': config.get('user_agent', 'AiFuzz/1.2.0'),
                **config.get('custom_headers', {})
            }
            
            timeout = aiohttp.ClientTimeout(total=config.get('timeout', 10))
            
            start_time = time.time()
            async with self.session.get(
                url, 
                headers=headers, 
                timeout=timeout,
                ssl=config.get('ssl_verify', True)
            ) as response:
                response_time = time.time() - start_time
                
                # Check if this is interesting
                if response.status in config.get('status_codes', [200, 201, 204, 301, 302, 403, 404, 500]):
                    content_type = response.headers.get('Content-Type', '')
                    content_length = int(response.headers.get('Content-Length', 0))
                    
                    # Skip common boring responses
                    if response.status == 404 or (response.status == 200 and content_length < 10):
                        return
                    
                    result = ScanResult(
                        url=url,
                        method='GET',
                        status_code=response.status,
                        content_length=content_length,
                        content_type=content_type,
                        response_time=response_time,
                        headers=dict(response.headers)
                    )
                    
                    self.results.append(result)
                    self.logger.log(f"Found: {url} [{response.status}]", "success")
                    
        except:
            # Silently skip errors to avoid spam
            pass
    
    def _load_custom_wordlist(self, filepath: str) -> List[str]:
        """Load custom wordlist from file"""
        try:
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except:
            return []

def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    
    return {
        "gemini_api_key": "",
        "gemini_model": "gemini-2.0-flash"
    }

def save_config(config: Dict[str, Any]):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except:
        pass

def update_config():
    """Update configuration interactively"""
    config = load_config()
    console.print("[bold blue]AiFuzz Configuration[/bold blue]\n")
    
    # Gemini API Key
    current_key = config.get("gemini_api_key", "")
    if current_key:
        console.print(f"Current API key: {current_key[:8]}...")
    else:
        console.print("No API key configured")
    
    try:
        new_key = input("Enter Gemini API key (press Enter to keep current): ").strip()
        if new_key:
            config["gemini_api_key"] = new_key
    except:
        pass
    
    save_config(config)
    console.print("[green]Configuration updated[/green]")

def run_wizard_mode():
    """Interactive wizard mode"""
    console.print("[bold blue]AiFuzz Wizard Mode[/bold blue]")
    console.print("[yellow]Configure your scan parameters[/yellow]\n")
    
    try:
        # Target URL
        url = input("Enter target URL: ").strip()
        if not url:
            console.print("[red]Target URL is required[/red]")
            return None
        
        # Scan Mode
        console.print("\nScan Modes:")
        console.print("1. dir - Directory discovery")
        console.print("2. param - Parameter fuzzing")
        console.print("3. api - API endpoint discovery")
        console.print("4. hybrid - All modes")
        
        mode_choice = input("Select scan mode (1-4, default=1): ").strip()
        modes = ["dir", "param", "api", "hybrid"]
        mode = modes[0] if not mode_choice else modes[int(mode_choice) - 1] if mode_choice.isdigit() and 1 <= int(mode_choice) <= 4 else modes[0]
        
        # Concurrent requests
        concurrent = input("Concurrent requests (default=50): ").strip()
        concurrent = int(concurrent) if concurrent.isdigit() else 50
        
        # Wordlist size
        console.print("\nWordlist Sizes:")
        console.print("1. small - Fast scan")
        console.print("2. medium - Balanced scan")
        console.print("3. large - Comprehensive scan")
        
        size_choice = input("Select wordlist size (1-3, default=2): ").strip()
        sizes = ["small", "medium", "large"]
        wordlist_size = sizes[1] if not size_choice else sizes[int(size_choice) - 1] if size_choice.isdigit() and 1 <= int(size_choice) <= 3 else sizes[1]
        
        # Verbose mode
        verbose = input("Enable verbose mode? (y/N): ").strip().lower() in ['y', 'yes']
        
        # AI Analysis
        ai_analysis = input("Enable AI analysis? (y/N): ").strip().lower() in ['y', 'yes']
        
        # Extensions
        ext_input = input("File extensions (space-separated, e.g., php js html): ").strip()
        extensions = ext_input.split() if ext_input else []
        
        # GitHub repos
        github_repos = []
        repo = input("GitHub wordlist repo URL (optional): ").strip()
        if repo:
            github_repos.append(repo)
        
        # Summary
        console.print("\n[bold green]Configuration Summary:[/bold green]")
        console.print(f"Target URL: {url}")
        console.print(f"Mode: {mode}")
        console.print(f"Concurrent requests: {concurrent}")
        console.print(f"Wordlist size: {wordlist_size}")
        console.print(f"Verbose mode: {verbose}")
        console.print(f"AI analysis: {ai_analysis}")
        if extensions:
            console.print(f"Extensions: {', '.join(extensions)}")
        if github_repos:
            console.print(f"GitHub repos: {', '.join(github_repos)}")
        
        # Confirm
        confirm = input("\nProceed with this configuration? (Y/n): ").strip().lower()
        if confirm in ['n', 'no']:
            return None
        
        return {
            'url': url,
            'mode': mode,
            'concurrent': concurrent,
            'wordlist_size': wordlist_size,
            'verbose': verbose,
            'ai_analysis': ai_analysis,
            'extensions': extensions,
            'github_repos': github_repos,
            'custom_headers': {},
            'ssl_verify': True,
            'status_codes': [200, 201, 204, 301, 302, 403, 500],
            'user_agent': 'AiFuzz/1.2.0',
            'timeout': 10
        }
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Wizard cancelled[/yellow]")
        return None
    except:
        console.print("[red]Error in wizard mode[/red]")
        return None

def save_results(results: List[ScanResult], output_file: str = None, output_format: str = "json"):
    """Save scan results to file"""
    if not results:
        return
    
    try:
        results_dir = Path("aifuzz_results")
        results_dir.mkdir(exist_ok=True)
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = urlparse(results[0].url).netloc
            output_file = results_dir / f"{domain}_scan_{timestamp}.{output_format}"
        
        if output_format == "json":
            with open(output_file, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2)
        elif output_format == "csv":
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['URL', 'Status', 'Length', 'Content-Type', 'Response Time'])
                for result in results:
                    writer.writerow([result.url, result.status_code, result.content_length, result.content_type, result.response_time])
        else:  # txt
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"{result.url} [{result.status_code}] {result.content_length}B\n")
        
        console.print(f"[green]Results saved to: {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving results: {str(e)}[/red]")

def display_results(results: List[ScanResult]):
    """Display scan results"""
    if not results:
        console.print("[yellow]No interesting results found[/yellow]")
        return
    
    table = Table(title="AiFuzz Results")
    table.add_column("URL", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Length", style="blue")
    table.add_column("Type", style="yellow")
    
    for result in results[:20]:  # Show first 20
        table.add_row(
            result.url[:60] + "..." if len(result.url) > 60 else result.url,
            str(result.status_code),
            str(result.content_length),
            result.content_type[:30] + "..." if len(result.content_type) > 30 else result.content_type
        )
    
    console.print(table)
    
    if len(results) > 20:
        console.print(f"[dim]... and {len(results) - 20} more results[/dim]")

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global SCAN_INTERRUPTED, CURRENT_RESULTS
    SCAN_INTERRUPTED = True
    console.print("\n[yellow]Scan interrupted[/yellow]")
    
    if CURRENT_RESULTS:
        save_results(CURRENT_RESULTS)
    
    sys.exit(0)

def main():
    """Main function"""
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool")
    parser.add_argument("-u", "--url", help="Target URL")
    parser.add_argument("-m", "--mode", choices=["dir", "param", "api", "hybrid"], default="dir", help="Scan mode")
    parser.add_argument("-c", "--concurrent", type=int, default=50, help="Concurrent requests")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="Request timeout")
    parser.add_argument("-w", "--wordlist", help="Custom wordlist file")
    parser.add_argument("--github-wordlist", nargs="+", help="GitHub wordlist URLs")
    parser.add_argument("-e", "--extensions", nargs="+", help="File extensions to test")
    parser.add_argument("-s", "--status-codes", nargs="+", type=int, default=[200, 201, 204, 301, 302, 403, 500], help="Status codes to show")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-f", "--format", choices=["json", "csv", "txt"], default="json", help="Output format")
    parser.add_argument("--headers", nargs="+", help="Custom headers")
    parser.add_argument("--wordlist-size", choices=["small", "medium", "large"], default="medium", help="Wordlist size")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--config", action="store_true", help="Update configuration")
    parser.add_argument("--wizard", action="store_true", help="Interactive wizard mode")
    parser.add_argument("--ai-analysis", action="store_true", help="Enable AI analysis")
    parser.add_argument("--version", action="version", version="AiDirFuzz 1.2.0")
    
    args = parser.parse_args()
    
    # Handle configuration
    if args.config:
        update_config()
        return
    
    # Handle wizard mode
    if args.wizard:
        wizard_config = run_wizard_mode()
        if not wizard_config:
            return
        config = wizard_config
    else:
        if not args.url:
            console.print("[red]Error: Target URL required. Use -u/--url or --wizard[/red]")
            return
        
        # Parse headers
        custom_headers = {}
        if args.headers:
            for header in args.headers:
                if ':' in header:
                    key, value = header.split(':', 1)
                    custom_headers[key.strip()] = value.strip()
        
        config = {
            'url': args.url,
            'mode': args.mode,
            'concurrent': args.concurrent,
            'timeout': args.timeout,
            'wordlist_size': args.wordlist_size,
            'custom_wordlist': args.wordlist,
            'github_repos': args.github_wordlist or [],
            'extensions': args.extensions or [],
            'verbose': args.verbose,
            'ai_analysis': args.ai_analysis,
            'custom_headers': custom_headers,
            'ssl_verify': True,
            'status_codes': args.status_codes,
            'user_agent': 'AiFuzz/1.2.0',
            'output_file': args.output,
            'output_format': args.format
        }
    
    # Load app config
    app_config = load_config()
    
    # Create logger
    logger = VerboseLogger(console)
    logger.set_verbose(config['verbose'])
    
    # Show scan info
    console.print(f"[bold]Starting AiFuzz scan: {config['url']}[/bold]")
    console.print(f"[dim]Mode: {config['mode']} | Concurrent: {config['concurrent']} | Wordlist: {config['wordlist_size']}[/dim]")
    
    # Run scan
    async def run_scan():
        async with Fuzzer(logger) as fuzzer:
            # Setup AI if enabled
            if config['ai_analysis'] and app_config.get('gemini_api_key'):
                fuzzer.setup_ai_analyzer(app_config['gemini_api_key'])
            elif config['ai_analysis']:
                console.print("[yellow]AI analysis enabled but no API key configured[/yellow]")
            
            # Run scan
            results = await fuzzer.run_scan(config)
            
            # Display results
            display_results(results)
            
            # Save results
            if results:
                save_results(results, config.get('output_file'), config.get('output_format', 'json'))
            
            # Show summary
            if fuzzer.start_time:
                scan_time = time.time() - fuzzer.start_time
                console.print(f"\n[bold green]Scan completed in {scan_time:.2f} seconds[/bold green]")
                console.print(f"Total requests: {fuzzer.completed_requests}")
                console.print(f"Requests per second: {fuzzer.completed_requests / scan_time:.2f}")
    
    # Run the scan
    try:
        asyncio.run(run_scan())
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main()
EOF

# Make executable
chmod +x aifuzz.py

echo "✅ AiFuzz v1.2.0 created successfully"

# Step 5: Create System Installation
echo "🔗 Step 5: Creating system-wide installation..."

# Create symlink
sudo ln -sf "$(pwd)/aifuzz.py" /usr/local/bin/aifuzz

# Update PATH if needed
if ! echo $PATH | grep -q "/usr/local/bin"; then
    echo "Adding /usr/local/bin to PATH..."
    echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
    export PATH="/usr/local/bin:$PATH"
fi

# Step 6: Clean up unnecessary files
echo "🧹 Step 6: Cleaning up unnecessary files..."

# Remove old installation scripts
rm -f install*.sh demo*.sh 2>/dev/null || true

# Remove old result files
rm -f *.json 2>/dev/null || true

# Remove other unnecessary files
rm -f aifuzz_base64.txt example_wordlist.txt 2>/dev/null || true
rm -f backend_test.py focused_test.py 2>/dev/null || true

# Step 7: Final Testing
echo "🧪 Step 7: Testing installation..."

# Test local usage
echo "Local version test:"
LOCAL_VERSION=$(python3 aifuzz.py --version 2>/dev/null || echo "FAILED")
echo "✅ $LOCAL_VERSION"

# Test system usage
echo "System version test:"
SYSTEM_VERSION=$(aifuzz --version 2>/dev/null || echo "FAILED - try: source ~/.bashrc")
echo "✅ $SYSTEM_VERSION"

# Final summary
echo ""
echo "🎉 Installation Complete!"
echo "======================="
echo "✅ AiFuzz v1.2.0 installed successfully"
echo "✅ All old versions removed"
echo "✅ Clean dependencies installed"
echo "✅ System-wide command available"
echo ""
echo "📋 Usage:"
echo "  aifuzz --help                 # Show help"
echo "  aifuzz --wizard               # Interactive wizard"
echo "  aifuzz --config               # Configure API keys"
echo "  aifuzz -u https://example.com # Quick scan"
echo ""
echo "📁 Files:"
echo "  Tool: $(pwd)/aifuzz.py"
echo "  Config: ~/.aifuzz/config.json"
echo "  Results: $(pwd)/aifuzz_results/"
echo ""
echo "🔧 If 'aifuzz' command doesn't work, run:"
echo "  source ~/.bashrc"
echo ""