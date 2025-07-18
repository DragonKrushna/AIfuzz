#!/usr/bin/env python3

"""
AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool
Created by: AI Development Team
Version: 1.2.0

A comprehensive penetration testing tool that combines traditional directory busting
with AI-powered analysis for smarter, more efficient security testing.

Key Improvements:
- Fixed wizard mode parameter passing
- Improved GitHub wordlist fetching with timeout and chunking
- Direct Gemini API integration
- Enhanced progress tracking and verbose mode
- Better error handling and recovery
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
from urllib.parse import urljoin, urlparse, parse_qs, urlencode
import base64
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import threading
import signal
import select
import tty
import termios

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich import print as rprint
    from rich.status import Status
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

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)

# Default wordlist configurations with better limits
WORDLIST_CONFIGS = {
    "small": {
        "dir": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/common.txt", 500),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/big.txt", 300),
        ],
        "param": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/burp-parameter-names.txt", 400),
            ("danielmiessler/SecLists/master/Fuzzing/LFI/LFI-Jhaddix.txt", 200),
        ],
        "api": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/api/api-endpoints.txt", 300),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/swagger.txt", 200),
        ]
    },
    "medium": {
        "dir": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/common.txt", 1000),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/big.txt", 800),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/directory-list-2.3-medium.txt", 1200),
        ],
        "param": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/burp-parameter-names.txt", 800),
            ("danielmiessler/SecLists/master/Fuzzing/LFI/LFI-Jhaddix.txt", 400),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/api/objects.txt", 600),
        ],
        "api": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/api/api-endpoints.txt", 600),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/swagger.txt", 400),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/graphql.txt", 300),
        ]
    },
    "large": {
        "dir": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/common.txt", 2000),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/big.txt", 1500),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/directory-list-2.3-medium.txt", 2500),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/raft-large-directories.txt", 1000),
        ],
        "param": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/burp-parameter-names.txt", 1500),
            ("danielmiessler/SecLists/master/Fuzzing/LFI/LFI-Jhaddix.txt", 800),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/api/objects.txt", 1200),
        ],
        "api": [
            ("danielmiessler/SecLists/master/Discovery/Web-Content/api/api-endpoints.txt", 1200),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/swagger.txt", 800),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/graphql.txt", 600),
            ("danielmiessler/SecLists/master/Discovery/Web-Content/api/actions.txt", 400),
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
    ai_analysis: str = ""
    ai_score: float = 0.0
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}

class VerboseLogger:
    """Handles verbose logging and interactive mode with improved consistency"""
    
    def __init__(self, console: Console):
        self.console = console
        self.is_verbose = False
        self.is_interactive_verbose = False
        self.verbose_timer = None
        self.logs = []
        self.max_logs = 100
        
    def log(self, message: str, level: str = "info"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)
        
        # Always show if verbose is enabled (either permanent or interactive)
        if self.is_verbose or self.is_interactive_verbose:
            if level == "error":
                self.console.print(f"[red]{log_entry}[/red]")
            elif level == "warning":
                self.console.print(f"[yellow]{log_entry}[/yellow]")
            elif level == "success":
                self.console.print(f"[green]{log_entry}[/green]")
            else:
                self.console.print(f"[dim]{log_entry}[/dim]")
    
    def toggle_interactive_verbose(self):
        """Toggle interactive verbose mode with improved timer"""
        if not self.is_interactive_verbose:
            self.is_interactive_verbose = True
            self.log("Interactive verbose mode enabled for 4 seconds", "info")
            
            # Show recent logs
            for log in self.logs[-10:]:  # Show last 10 logs
                self.console.print(f"[dim]{log}[/dim]")
            
            # Set timer to disable verbose mode
            if self.verbose_timer:
                self.verbose_timer.cancel()
            self.verbose_timer = threading.Timer(4.0, self.disable_interactive_verbose)
            self.verbose_timer.start()
        else:
            self.disable_interactive_verbose()
    
    def disable_interactive_verbose(self):
        """Disable interactive verbose mode"""
        if self.verbose_timer:
            self.verbose_timer.cancel()
        self.is_interactive_verbose = False
        self.log("Interactive verbose mode disabled, returning to progress bar", "info")
    
    def set_verbose(self, enabled: bool):
        """Set permanent verbose mode"""
        self.is_verbose = enabled
        if enabled:
            self.log("Verbose mode enabled", "info")

class ImprovedWordlistFetcher:
    """Improved wordlist fetcher with better timeout handling and chunking"""
    
    def __init__(self, verbose_logger: VerboseLogger):
        self.session = None
        self.cache = {}
        self.logger = verbose_logger
        self.custom_repos = []
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)  # 30s total, 10s connect
    
    def add_custom_repo(self, repo_url: str):
        """Add custom GitHub repository URL"""
        self.custom_repos.append(repo_url)
        self.logger.log(f"Added custom wordlist repo: {repo_url}")
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={'User-Agent': 'AiFuzz/1.2.0 (Security Scanner)'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_wordlist_chunked(self, repo_path: str, limit: int = -1, chunk_size: int = 8192) -> List[str]:
        """Fetch wordlist with chunked reading to avoid memory issues"""
        cache_key = f"{repo_path}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            self.logger.log(f"Fetching wordlist from {repo_path}")
            # Convert GitHub blob URL to raw content URL
            if repo_path.startswith("http"):
                raw_url = repo_path
            else:
                raw_url = f"https://raw.githubusercontent.com/{repo_path.replace('/blob/', '/')}"
            
            wordlist = []
            async with self.session.get(raw_url) as response:
                if response.status == 200:
                    content_length = response.headers.get('content-length')
                    if content_length:
                        total_size = int(content_length)
                        self.logger.log(f"Downloading {total_size} bytes from {repo_path}")
                    
                    content = ""
                    async for chunk in response.content.iter_chunked(chunk_size):
                        content += chunk.decode('utf-8', errors='ignore')
                        
                        # Process lines as we go to avoid memory issues
                        while '\n' in content:
                            line, content = content.split('\n', 1)
                            line = line.strip()
                            if line and not line.startswith('#'):
                                wordlist.append(line)
                                
                                # Apply limit early to avoid processing too much
                                if limit > 0 and len(wordlist) >= limit:
                                    self.logger.log(f"Reached limit of {limit} words from {repo_path}")
                                    break
                        
                        # Check if we should stop
                        if limit > 0 and len(wordlist) >= limit:
                            break
                    
                    # Process any remaining content
                    if content.strip() and (limit <= 0 or len(wordlist) < limit):
                        line = content.strip()
                        if line and not line.startswith('#'):
                            wordlist.append(line)
                    
                    # Apply final limit
                    if limit > 0:
                        wordlist = wordlist[:limit]
                    
                    self.cache[cache_key] = wordlist
                    self.logger.log(f"Loaded {len(wordlist)} words from {repo_path}")
                    return wordlist
                else:
                    self.logger.log(f"Failed to fetch wordlist from {repo_path}: {response.status}", "error")
                    return []
        except asyncio.TimeoutError:
            self.logger.log(f"Timeout fetching wordlist from {repo_path}", "error")
            return []
        except Exception as e:
            self.logger.log(f"Error fetching wordlist from {repo_path}: {str(e)}", "error")
            return []
    
    async def fetch_custom_wordlist(self, repo_url: str, limit: int = -1) -> List[str]:
        """Fetch wordlist from custom GitHub repository URL with improved error handling"""
        try:
            self.logger.log(f"Fetching custom wordlist from {repo_url}")
            
            # Convert GitHub URL to raw content URL
            if "github.com" in repo_url and "/blob/" in repo_url:
                raw_url = repo_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            else:
                raw_url = repo_url
            
            return await self.fetch_wordlist_chunked(raw_url, limit)
            
        except Exception as e:
            self.logger.log(f"Error fetching custom wordlist: {str(e)}", "error")
            return []
    
    async def get_combined_wordlist(self, wordlist_type: str, size: str = "medium") -> List[str]:
        """Get combined wordlist from multiple sources with progress tracking"""
        all_words = set()
        
        # Add words from default wordlists
        if size in WORDLIST_CONFIGS and wordlist_type in WORDLIST_CONFIGS[size]:
            self.logger.log(f"Loading {size} {wordlist_type} wordlist")
            for i, (repo_path, limit) in enumerate(WORDLIST_CONFIGS[size][wordlist_type]):
                self.logger.log(f"Fetching source {i+1}/{len(WORDLIST_CONFIGS[size][wordlist_type])}: {repo_path}")
                try:
                    words = await self.fetch_wordlist_chunked(repo_path, limit)
                    all_words.update(words)
                    self.logger.log(f"Added {len(words)} words from source {i+1}")
                except Exception as e:
                    self.logger.log(f"Failed to fetch from source {i+1}: {str(e)}", "error")
                    continue
        
        # Add words from custom repositories
        if self.custom_repos:
            self.logger.log(f"Loading from {len(self.custom_repos)} custom repositories")
            for i, repo_url in enumerate(self.custom_repos):
                self.logger.log(f"Fetching custom source {i+1}/{len(self.custom_repos)}: {repo_url}")
                try:
                    words = await self.fetch_custom_wordlist(repo_url)
                    all_words.update(words)
                    self.logger.log(f"Added {len(words)} words from custom source {i+1}")
                except Exception as e:
                    self.logger.log(f"Failed to fetch from custom source {i+1}: {str(e)}", "error")
                    continue
        
        result = list(all_words)
        self.logger.log(f"Combined {wordlist_type} wordlist: {len(result)} words")
        return result

class DirectGeminiAnalyzer:
    """Direct Gemini API integration without emergentintegrations"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", verbose_logger: VerboseLogger = None):
        self.api_key = api_key
        self.model = model
        self.logger = verbose_logger or VerboseLogger(console)
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup the Gemini client"""
        try:
            self.logger.log("Initializing Gemini client")
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.logger.log("Gemini client initialized successfully")
        except Exception as e:
            self.logger.log(f"Failed to initialize Gemini client: {str(e)}", "error")
            self.client = None
    
    async def analyze_scan_results(self, results: List[ScanResult], target_url: str) -> Dict[str, Any]:
        """Analyze scan results after scan completion"""
        if not self.client:
            return {"error": "Gemini client not initialized"}
        
        try:
            self.logger.log(f"Running post-scan AI analysis on {len(results)} results")
            
            # Prepare analysis data
            analysis_data = {
                "target_url": target_url,
                "total_results": len(results),
                "status_codes": {},
                "content_types": {},
                "interesting_findings": [],
                "potential_vulnerabilities": []
            }
            
            # Analyze status codes and content types
            for result in results:
                status = str(result.status_code)
                analysis_data["status_codes"][status] = analysis_data["status_codes"].get(status, 0) + 1
                
                if result.content_type:
                    ct = result.content_type.split(';')[0].strip()
                    analysis_data["content_types"][ct] = analysis_data["content_types"].get(ct, 0) + 1
                
                # Flag interesting findings
                if result.status_code in [200, 201, 204, 301, 302, 401, 403, 500]:
                    analysis_data["interesting_findings"].append({
                        "url": result.url,
                        "status": result.status_code,
                        "content_type": result.content_type,
                        "size": result.content_length
                    })
            
            # Prepare prompt for Gemini
            prompt = f"""
            Analyze the following web application scan results for security implications:

            Target: {target_url}
            Total endpoints found: {len(results)}
            Status code distribution: {analysis_data['status_codes']}
            Content type distribution: {analysis_data['content_types']}

            Key findings:
            {json.dumps(analysis_data['interesting_findings'][:10], indent=2)}

            Please provide:
            1. Security assessment summary
            2. Potential vulnerabilities identified
            3. Recommended next steps for penetration testing
            4. Risk score (1-10)
            5. Priority findings to investigate

            Return your analysis in JSON format.
            """
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            self.logger.log("AI analysis completed successfully")
            return {
                "analysis": response.text,
                "analysis_data": analysis_data,
                "error": None
            }
            
        except Exception as e:
            self.logger.log(f"AI analysis failed: {str(e)}", "error")
            return {"error": f"AI analysis failed: {str(e)}"}

class KeyboardHandler:
    """Handles keyboard input for interactive features"""
    
    def __init__(self, verbose_logger: VerboseLogger):
        self.logger = verbose_logger
        self.running = False
        self.thread = None
    
    def start(self):
        """Start keyboard listener"""
        self.running = True
        self.thread = threading.Thread(target=self._listen_for_input, daemon=True)
        self.thread.start()
        if not self.logger.is_verbose:
            self.logger.log("Keyboard listener started (Press Enter for verbose mode)")
    
    def stop(self):
        """Stop keyboard listener"""
        self.running = False
    
    def _listen_for_input(self):
        """Listen for keyboard input"""
        try:
            while self.running:
                if sys.stdin.isatty():
                    # Non-blocking input check
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        line = sys.stdin.readline()
                        if line.strip() == "":  # Enter key pressed
                            self.logger.toggle_interactive_verbose()
                else:
                    # In non-interactive environment, just wait
                    time.sleep(0.1)
        except Exception as e:
            self.logger.log(f"Keyboard handler error: {str(e)}", "error")

class ImprovedFuzzer:
    """Improved fuzzing engine with better progress tracking and error handling"""
    
    def __init__(self, verbose_logger: VerboseLogger):
        self.logger = verbose_logger
        self.session = None
        self.wordlist_fetcher = ImprovedWordlistFetcher(self.logger)
        self.ai_analyzer = None
        self.keyboard_handler = KeyboardHandler(self.logger)
        self.results = []
        self.progress_task = None
        self.progress = None
        self.completed_requests = 0
        self.total_requests = 0
        self.start_time = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def setup_ai_analyzer(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Setup AI analyzer with Gemini API"""
        if api_key:
            self.ai_analyzer = DirectGeminiAnalyzer(api_key, model, self.logger)
        else:
            self.logger.log("No API key provided, AI analysis disabled", "warning")
    
    async def run_scan(self, config: Dict[str, Any]) -> List[ScanResult]:
        """Run the fuzzing scan with improved progress tracking"""
        global CURRENT_RESULTS
        
        self.start_time = time.time()
        self.results = []
        CURRENT_RESULTS = self.results
        
        try:
            # Load wordlist
            console.print("[yellow]Fetching wordlists from GitHub...[/yellow]")
            
            # Setup wordlist fetcher
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
            
            # Setup keyboard handler
            self.keyboard_handler.start()
            
            # Create progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=10
            ) as progress:
                self.progress = progress
                self.progress_task = progress.add_task(
                    f"Scanning...", 
                    total=self.total_requests
                )
                
                # Run scan
                await self._execute_scan(wordlist, config)
            
            # Stop keyboard handler
            self.keyboard_handler.stop()
            
            # Post-scan AI analysis
            if self.ai_analyzer and self.results:
                console.print("[yellow]Running AI analysis on scan results...[/yellow]")
                ai_results = await self.ai_analyzer.analyze_scan_results(self.results, config['url'])
                if ai_results.get('error'):
                    console.print(f"[red]AI analysis failed: {ai_results['error']}[/red]")
                else:
                    console.print("[green]AI analysis completed[/green]")
                    # You could save AI results to file here
            
            return self.results
            
        except Exception as e:
            self.logger.log(f"Scan error: {str(e)}", "error")
            return self.results
    
    async def _execute_scan(self, wordlist: List[str], config: Dict[str, Any]):
        """Execute the actual scan with improved concurrency handling"""
        semaphore = asyncio.Semaphore(config.get('concurrent', 50))
        
        async def scan_word(word: str):
            async with semaphore:
                await self._scan_single_word(word, config)
                self.completed_requests += 1
                if self.progress:
                    self.progress.update(self.progress_task, completed=self.completed_requests)
        
        # Create tasks in batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(wordlist), batch_size):
            batch = wordlist[i:i + batch_size]
            tasks = [scan_word(word) for word in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Add delay between batches if configured
            if config.get('delay', 0) > 0:
                await asyncio.sleep(config['delay'])
    
    async def _scan_single_word(self, word: str, config: Dict[str, Any]):
        """Scan a single word with improved error handling"""
        try:
            target_url = config['url'].rstrip('/')
            
            if config['mode'] == 'dir':
                test_url = f"{target_url}/{word}"
            elif config['mode'] == 'param':
                test_url = f"{target_url}?{word}=test"
            elif config['mode'] == 'api':
                test_url = f"{target_url}/api/{word}"
            else:  # hybrid
                # Test as directory, parameter, and API
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
                
        except Exception as e:
            self.logger.log(f"Error scanning word '{word}': {str(e)}", "error")
    
    async def _test_url(self, url: str, config: Dict[str, Any]):
        """Test a single URL with improved error handling"""
        try:
            headers = {
                'User-Agent': config.get('user_agent', 'AiFuzz/1.2.0 (Security Scanner)'),
                **config.get('custom_headers', {})
            }
            
            timeout = aiohttp.ClientTimeout(total=config.get('timeout', 10))
            
            start_time = time.time()
            async with self.session.get(
                url, 
                headers=headers, 
                timeout=timeout,
                ssl=config.get('ssl_verify', True),
                proxy=config.get('proxy')
            ) as response:
                response_time = time.time() - start_time
                
                # Check if this is an interesting response
                if response.status in config.get('status_codes', [200, 201, 204, 301, 302, 307, 308, 403, 405, 500]):
                    content_type = response.headers.get('Content-Type', '')
                    content_length = int(response.headers.get('Content-Length', 0))
                    
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
                    self.logger.log(f"Found: {url} [{response.status}] {content_length}B {content_type}")
                    
        except asyncio.TimeoutError:
            self.logger.log(f"Timeout: {url}", "warning")
        except Exception as e:
            self.logger.log(f"Error testing {url}: {str(e)}", "error")
    
    def _load_custom_wordlist(self, filepath: str) -> List[str]:
        """Load custom wordlist from file"""
        try:
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            self.logger.log(f"Error loading custom wordlist: {str(e)}", "error")
            return []

def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading config: {str(e)}[/red]")
    
    # Return default config
    return {
        "gemini_api_key": "",
        "gemini_model": "gemini-2.0-flash",
        "default_concurrent": 50,
        "default_timeout": 10
    }

def save_config(config: Dict[str, Any]):
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        console.print(f"[green]Configuration saved to {CONFIG_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving config: {str(e)}[/red]")

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
    
    new_key = input("Enter Gemini API key (press Enter to keep current): ").strip()
    if new_key:
        config["gemini_api_key"] = new_key
    
    # Gemini Model
    console.print(f"\nCurrent model: {config.get('gemini_model', 'gemini-2.0-flash')}")
    
    print("Available Gemini models:")
    models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    try:
        model_choice = input(f"Select model (1-{len(models)}, press Enter for current): ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
            config["gemini_model"] = models[int(model_choice) - 1]
    except:
        pass
    
    save_config(config)

def run_wizard_mode():
    """Interactive wizard mode with improved parameter passing"""
    console.print("[bold blue]AiFuzz Wizard Mode[/bold blue]")
    console.print("[yellow]This wizard will help you configure your scan parameters[/yellow]\n")
    
    # Target URL
    url = input("Enter target URL: ").strip()
    if not url:
        console.print("[red]Target URL is required[/red]")
        return None
    
    # Scan Mode
    console.print("\n[bold]Scan Modes:[/bold]")
    modes = ["dir", "param", "api", "hybrid"]
    for i, mode in enumerate(modes, 1):
        descriptions = {
            "dir": "Directory and file discovery",
            "param": "Parameter fuzzing",
            "api": "API endpoint discovery",
            "hybrid": "All modes combined"
        }
        console.print(f"{i}. {mode} - {descriptions[mode]}")
    
    while True:
        try:
            mode_choice = input("Select scan mode (1-4, default=1): ").strip()
            if not mode_choice:
                mode_choice = "1"
            mode_idx = int(mode_choice) - 1
            if 0 <= mode_idx < len(modes):
                mode = modes[mode_idx]
                break
            else:
                console.print("[red]Invalid choice. Please select 1-4.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Concurrent requests
    while True:
        try:
            concurrent = input("Concurrent requests (default=50): ").strip()
            if not concurrent:
                concurrent = 50
            else:
                concurrent = int(concurrent)
            if concurrent > 0:
                break
            else:
                console.print("[red]Concurrent requests must be positive[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Timeout
    while True:
        try:
            timeout = input("Request timeout in seconds (default=10): ").strip()
            if not timeout:
                timeout = 10
            else:
                timeout = int(timeout)
            if timeout > 0:
                break
            else:
                console.print("[red]Timeout must be positive[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Delay
    while True:
        try:
            delay = input("Delay between requests in seconds (default=0.0): ").strip()
            if not delay:
                delay = 0.0
            else:
                delay = float(delay)
            if delay >= 0:
                break
            else:
                console.print("[red]Delay must be non-negative[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Wordlist size
    console.print("\n[bold]Wordlist Sizes:[/bold]")
    sizes = ["small", "medium", "large"]
    for i, size in enumerate(sizes, 1):
        descriptions = {
            "small": "Fast scan (~1,500 words)",
            "medium": "Balanced scan (~3,000 words)",
            "large": "Comprehensive scan (~6,000 words)"
        }
        console.print(f"{i}. {size} - {descriptions[size]}")
    
    while True:
        try:
            size_choice = input("Select wordlist size (1-3, default=2): ").strip()
            if not size_choice:
                size_choice = "2"
            size_idx = int(size_choice) - 1
            if 0 <= size_idx < len(sizes):
                wordlist_size = sizes[size_idx]
                break
            else:
                console.print("[red]Invalid choice. Please select 1-3.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Custom wordlist
    custom_wordlist = input("\nCustom wordlist file path (optional): ").strip()
    if custom_wordlist and not os.path.exists(custom_wordlist):
        console.print(f"[yellow]Warning: Custom wordlist file not found: {custom_wordlist}[/yellow]")
        custom_wordlist = None
    
    # GitHub wordlist repos
    github_repos = []
    while True:
        repo = input("GitHub wordlist repo URL (optional, press Enter to skip): ").strip()
        if not repo:
            break
        github_repos.append(repo)
        console.print(f"[green]Added: {repo}[/green]")
    
    # Extensions
    extensions = []
    ext_input = input("File extensions to test (space-separated, e.g., php js html): ").strip()
    if ext_input:
        extensions = ext_input.split()
    
    # Output format
    console.print("\n[bold]Output Formats:[/bold]")
    formats = ["json", "csv", "txt"]
    for i, fmt in enumerate(formats, 1):
        console.print(f"{i}. {fmt}")
    
    while True:
        try:
            format_choice = input("Select output format (1-3, default=1): ").strip()
            if not format_choice:
                format_choice = "1"
            format_idx = int(format_choice) - 1
            if 0 <= format_idx < len(formats):
                output_format = formats[format_idx]
                break
            else:
                console.print("[red]Invalid choice. Please select 1-3.[/red]")
        except ValueError:
            console.print("[red]Invalid input. Please enter a number.[/red]")
    
    # Custom output file
    output_file = input("Custom output file path (optional): ").strip()
    
    # Verbose mode
    verbose = input("Enable verbose mode? (y/N): ").strip().lower() in ['y', 'yes', '1']
    
    # AI Analysis
    ai_analysis = input("Enable AI analysis? (y/N): ").strip().lower() in ['y', 'yes', '1']
    
    # Custom headers
    custom_headers = {}
    console.print("\n[bold]Custom Headers (optional):[/bold]")
    console.print("[dim]Format: Header-Name:Header-Value[/dim]")
    while True:
        header = input("Enter custom header (press Enter to skip): ").strip()
        if not header:
            break
        if ":" in header:
            key, value = header.split(":", 1)
            custom_headers[key.strip()] = value.strip()
            console.print(f"[green]Added header: {key.strip()} = {value.strip()}[/green]")
        else:
            console.print("[red]Invalid format. Use Header-Name:Header-Value[/red]")
    
    # Proxy
    proxy = input("Proxy URL (optional): ").strip()
    
    # SSL verification
    ssl_verify = not (input("Disable SSL verification? (y/N): ").strip().lower() in ['y', 'yes', '1'])
    
    # Summary
    console.print("\n[bold green]Configuration Summary:[/bold green]")
    console.print(f"Target URL: {url}")
    console.print(f"Mode: {mode}")
    console.print(f"Concurrent requests: {concurrent}")
    console.print(f"Timeout: {timeout}s")
    console.print(f"Delay: {delay}s")
    console.print(f"Wordlist size: {wordlist_size}")
    if custom_wordlist:
        console.print(f"Custom wordlist: {custom_wordlist}")
    if github_repos:
        console.print(f"GitHub repos: {', '.join(github_repos)}")
    if extensions:
        console.print(f"Extensions: {', '.join(extensions)}")
    console.print(f"Output format: {output_format}")
    if output_file:
        console.print(f"Output file: {output_file}")
    console.print(f"Verbose mode: {verbose}")
    console.print(f"AI analysis: {ai_analysis}")
    if custom_headers:
        console.print(f"Custom headers: {len(custom_headers)} headers")
    if proxy:
        console.print(f"Proxy: {proxy}")
    console.print(f"SSL verification: {ssl_verify}")
    
    # Confirm
    if not input("\nProceed with this configuration? (Y/n): ").strip().lower() in ['n', 'no', '0']:
        return {
            'url': url,
            'mode': mode,
            'concurrent': concurrent,
            'timeout': timeout,
            'delay': delay,
            'wordlist_size': wordlist_size,
            'custom_wordlist': custom_wordlist,
            'github_repos': github_repos,
            'extensions': extensions,
            'output_format': output_format,
            'output_file': output_file,
            'verbose': verbose,
            'ai_analysis': ai_analysis,
            'custom_headers': custom_headers,
            'proxy': proxy,
            'ssl_verify': ssl_verify,
            'status_codes': [200, 201, 204, 301, 302, 307, 308, 403, 405, 500],
            'user_agent': 'AiFuzz/1.2.0 (Security Scanner)'
        }
    else:
        console.print("[yellow]Configuration cancelled[/yellow]")
        return None

def save_results(results: List[ScanResult], output_file: str, output_format: str):
    """Save scan results to file"""
    try:
        # Create results directory
        results_dir = Path("aifuzz_results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate filename if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            domain = urlparse(results[0].url if results else "unknown").netloc
            output_file = results_dir / f"{domain}_scan_{timestamp}.{output_format}"
        else:
            output_file = Path(output_file)
        
        # Save results
        if output_format == "json":
            with open(output_file, 'w') as f:
                json.dump([asdict(result) for result in results], f, indent=2)
        elif output_format == "csv":
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['URL', 'Method', 'Status', 'Length', 'Content-Type', 'Response Time', 'AI Analysis', 'AI Score'])
                for result in results:
                    writer.writerow([
                        result.url, result.method, result.status_code, 
                        result.content_length, result.content_type, 
                        result.response_time, result.ai_analysis, result.ai_score
                    ])
        else:  # txt
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"{result.url} [{result.status_code}] {result.content_length}B {result.content_type}\n")
        
        console.print(f"[green]Results saved to: {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error saving results: {str(e)}[/red]")

def display_results(results: List[ScanResult]):
    """Display scan results in a formatted table"""
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    table = Table(title="AiDirFuzz Results")
    table.add_column("URL", style="cyan", no_wrap=True)
    table.add_column("Method", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Length", style="blue")
    table.add_column("Type", style="yellow")
    table.add_column("Score", style="red")
    table.add_column("AI Analysis", style="bright_black")
    
    for result in results[:20]:  # Show first 20 results
        table.add_row(
            result.url[:50] + "..." if len(result.url) > 50 else result.url,
            result.method,
            str(result.status_code),
            str(result.content_length),
            result.content_type[:20] + "..." if len(result.content_type) > 20 else result.content_type,
            f"{result.ai_score:.1f}",
            result.ai_analysis[:20] + "..." if len(result.ai_analysis) > 20 else result.ai_analysis
        )
    
    console.print(table)
    
    if len(results) > 20:
        console.print(f"[dim]... and {len(results) - 20} more results[/dim]")

def signal_handler(signum, frame):
    """Handle interrupt signal"""
    global SCAN_INTERRUPTED, CURRENT_RESULTS
    SCAN_INTERRUPTED = True
    console.print("\n[yellow]Scan interrupted by user[/yellow]")
    
    if CURRENT_RESULTS:
        console.print("[yellow]Saving results from interrupted scan...[/yellow]")
        save_results(CURRENT_RESULTS, "", "json")
    
    sys.exit(0)

def main():
    """Main function with improved wizard mode integration"""
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aifuzz -u https://example.com -m dir -c 100
  aifuzz -u https://api.example.com -m api -c 50 -o custom_results.json
  aifuzz -u https://example.com -m param -w custom_params.txt
  aifuzz -u https://example.com -m hybrid -c 200 --ai-analysis
  aifuzz -u https://example.com -m dir --wordlist-size small -v
  aifuzz --wizard  # Interactive wizard mode
  aifuzz --config  # Update configuration

Results are automatically saved to aifuzz_results/ folder with filename format:
  domain_mode_timestamp.format (e.g., example.com_dir_20250107_143022.json)
        """
    )
    
    parser.add_argument("-u", "--url", help="Target URL")
    parser.add_argument("-m", "--mode", choices=["dir", "param", "api", "hybrid"], 
                       default="dir", help="Scan mode (default: dir)")
    parser.add_argument("-c", "--concurrent", type=int, default=50, 
                       help="Concurrent requests (default: 50)")
    parser.add_argument("-t", "--timeout", type=int, default=10, 
                       help="Request timeout in seconds (default: 10)")
    parser.add_argument("-d", "--delay", type=float, default=0.0, 
                       help="Delay between requests in seconds (default: 0.0)")
    parser.add_argument("-w", "--wordlist", help="Custom wordlist file")
    parser.add_argument("--github-wordlist", nargs="+", 
                       help="GitHub wordlist repository URLs")
    parser.add_argument("-e", "--extensions", nargs="+", 
                       help="File extensions to test (e.g., php js html)")
    parser.add_argument("-s", "--status-codes", nargs="+", type=int, 
                       default=[200, 201, 204, 301, 302, 307, 308, 403, 405, 500],
                       help="Status codes to show (default: 200 201 204 301 302 307 308 403 405 500)")
    parser.add_argument("-o", "--output", help="Output file path (default: auto-generated in aifuzz_results/)")
    parser.add_argument("-f", "--format", choices=["json", "csv", "txt"], 
                       default="json", help="Output format (default: json)")
    parser.add_argument("--headers", nargs="+", help="Custom headers (format: 'Header:Value')")
    parser.add_argument("--cookies", nargs="+", help="Custom cookies (format: 'name=value')")
    parser.add_argument("--proxy", help="Proxy URL (http://proxy:port)")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification")
    parser.add_argument("--ai-analysis", action="store_true", help="Enable AI analysis")
    parser.add_argument("--wordlist-size", choices=["small", "medium", "large"], 
                       default="medium", help="Wordlist size from GitHub (default: medium)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--config", action="store_true", help="Update configuration")
    parser.add_argument("--wizard", action="store_true", help="Interactive wizard mode")
    parser.add_argument("--user-agent", default="AiFuzz/1.2.0 (Security Scanner)", 
                       help="Custom User-Agent")
    parser.add_argument("--version", action="version", version="AiDirFuzz 1.2.0")
    
    args = parser.parse_args()
    
    # Handle configuration update
    if args.config:
        update_config()
        return
    
    # Handle wizard mode
    if args.wizard:
        wizard_config = run_wizard_mode()
        if not wizard_config:
            return
        
        # Use wizard configuration
        config = wizard_config
        console.print(f"\n[bold green]Starting scan with wizard configuration...[/bold green]")
    else:
        # Use command line arguments
        if not args.url:
            console.print("[red]Error: Target URL is required. Use -u/--url or --wizard mode.[/red]")
            return
        
        # Parse custom headers
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
            'delay': args.delay,
            'wordlist_size': args.wordlist_size,
            'custom_wordlist': args.wordlist,
            'github_repos': args.github_wordlist or [],
            'extensions': args.extensions or [],
            'output_format': args.format,
            'output_file': args.output,
            'verbose': args.verbose,
            'ai_analysis': args.ai_analysis,
            'custom_headers': custom_headers,
            'proxy': args.proxy,
            'ssl_verify': not args.no_ssl_verify,
            'status_codes': args.status_codes,
            'user_agent': args.user_agent
        }
    
    # Load configuration
    app_config = load_config()
    
    # Create logger
    logger = VerboseLogger(console)
    logger.set_verbose(config['verbose'])
    
    # Show scan info
    console.print(f"[bold]Starting AiDirFuzz scan against: {config['url']}[/bold]")
    console.print(f"[dim]Mode: {config['mode']} | Concurrent: {config['concurrent']} | Wordlist: {config['wordlist_size']}[/dim]")
    if not config['verbose']:
        console.print(f"[yellow]Press Enter during scan to enable verbose mode[/yellow]")
    
    # Run scan
    async def run_scan():
        async with ImprovedFuzzer(logger) as fuzzer:
            # Setup AI analyzer if enabled
            if config['ai_analysis'] and app_config.get('gemini_api_key'):
                fuzzer.setup_ai_analyzer(app_config['gemini_api_key'], app_config.get('gemini_model', 'gemini-2.0-flash'))
            elif config['ai_analysis']:
                console.print("[yellow]AI analysis enabled but no API key configured. Run --config to set up.[/yellow]")
            
            # Run the scan
            results = await fuzzer.run_scan(config)
            
            # Display results
            display_results(results)
            
            # Save results
            if results:
                save_results(results, config['output_file'], config['output_format'])
            
            # Show summary
            scan_time = time.time() - fuzzer.start_time
            console.print(f"\n[bold green]Scan completed in {scan_time:.2f} seconds[/bold green]")
            console.print(f"Total requests: {fuzzer.completed_requests}")
            console.print(f"Requests per second: {fuzzer.completed_requests / scan_time:.2f}")
            
            if not results:
                console.print("[yellow]No interesting results found[/yellow]")
    
    # Run the scan
    try:
        asyncio.run(run_scan())
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main()