#!/usr/bin/env python3

"""
AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool
Created by: AI Development Team
Version: 1.0.0

A comprehensive penetration testing tool that combines traditional directory busting
with AI-powered analysis for smarter, more efficient security testing.
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

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.align import Align
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Rich library not available. Installing...")
    os.system("pip install rich")
    try:
        from rich.console import Console
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        from rich.live import Live
        from rich.layout import Layout
        from rich.align import Align
        from rich import print as rprint
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False

# Initialize console
console = Console() if RICH_AVAILABLE else None

# Global variables
STOP_SCANNING = False
GITHUB_API_BASE = "https://api.github.com"
COMMON_WORDLISTS = {
    "directories": [
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/directory-list-2.3-medium.txt",
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/common.txt",
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/big.txt"
    ],
    "files": [
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/web-extensions.txt",
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/Common-DB-Backups.txt",
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/Common-PHP-Filenames.txt"
    ],
    "parameters": [
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/burp-parameter-names.txt",
        "danielmiessler/SecLists/blob/master/Fuzzing/LFI/LFI-Jhaddix.txt"
    ],
    "api": [
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/api/api-endpoints.txt",
        "danielmiessler/SecLists/blob/master/Discovery/Web-Content/swagger.txt"
    ]
}

@dataclass
class ScanResult:
    """Data class for scan results"""
    url: str
    method: str
    status_code: int
    content_length: int
    content_type: str
    response_time: float
    redirect_url: Optional[str] = None
    ai_analysis: Optional[str] = None
    vulnerability_score: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ScanConfig:
    """Configuration for scanning"""
    target_url: str
    mode: str
    concurrent_requests: int = 50
    timeout: int = 10
    delay: float = 0.0
    user_agent: str = "AiDirFuzz/1.0"
    custom_headers: Dict[str, str] = None
    cookies: Dict[str, str] = None
    proxy: Optional[str] = None
    follow_redirects: bool = True
    max_redirects: int = 5
    verify_ssl: bool = True
    custom_wordlist: Optional[str] = None
    extensions: List[str] = None
    status_codes: List[int] = None
    content_length_filter: Optional[int] = None
    output_format: str = "json"
    output_file: Optional[str] = None
    resume_file: Optional[str] = None
    ai_analysis: bool = True
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    
    def __post_init__(self):
        if self.custom_headers is None:
            self.custom_headers = {}
        if self.cookies is None:
            self.cookies = {}
        if self.extensions is None:
            self.extensions = []
        if self.status_codes is None:
            self.status_codes = [200, 201, 204, 301, 302, 307, 308, 403, 405, 500]

class GitHubWordlistFetcher:
    """Fetches wordlists from GitHub repositories without downloading"""
    
    def __init__(self):
        self.session = None
        self.cache = {}
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=10)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_wordlist(self, repo_path: str) -> List[str]:
        """Fetch wordlist from GitHub repository"""
        if repo_path in self.cache:
            return self.cache[repo_path]
        
        try:
            # Convert GitHub blob URL to raw content URL
            raw_url = f"https://raw.githubusercontent.com/{repo_path.replace('/blob/', '/')}"
            
            async with self.session.get(raw_url) as response:
                if response.status == 200:
                    content = await response.text()
                    # Filter out empty lines and comments
                    wordlist = [
                        line.strip() for line in content.split('\n') 
                        if line.strip() and not line.strip().startswith('#')
                    ]
                    self.cache[repo_path] = wordlist
                    return wordlist
                else:
                    console.print(f"[red]Failed to fetch wordlist from {repo_path}: {response.status}[/red]")
                    return []
        except Exception as e:
            console.print(f"[red]Error fetching wordlist from {repo_path}: {str(e)}[/red]")
            return []
    
    async def get_combined_wordlist(self, wordlist_type: str) -> List[str]:
        """Get combined wordlist from multiple sources"""
        all_words = set()
        
        if wordlist_type in COMMON_WORDLISTS:
            for repo_path in COMMON_WORDLISTS[wordlist_type]:
                words = await self.fetch_wordlist(repo_path)
                all_words.update(words)
        
        return list(all_words)

class AIAnalyzer:
    """AI-powered analysis using Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.session_id = str(uuid.uuid4())
        self.chat = None
        self._setup_chat()
    
    def _setup_chat(self):
        """Setup the AI chat instance"""
        try:
            from emergentintegrations.llm.chat import LlmChat, UserMessage
            self.chat = LlmChat(
                api_key=self.api_key,
                session_id=self.session_id,
                system_message="""You are an expert penetration testing AI assistant specializing in web application security analysis. Your role is to:

1. Analyze HTTP responses and identify potential security vulnerabilities
2. Generate intelligent wordlists based on target analysis
3. Filter false positives and identify interesting findings
4. Provide vulnerability scoring and recommendations
5. Suggest next steps for deeper analysis

Always provide concise, actionable insights focused on security implications."""
            ).with_model("gemini", self.model)
            
            self.UserMessage = UserMessage
            
        except Exception as e:
            console.print(f"[red]Failed to initialize AI chat: {str(e)}[/red]")
            self.chat = None
    
    async def analyze_target(self, target_url: str) -> Dict[str, Any]:
        """Analyze target URL and suggest scanning strategy"""
        if not self.chat:
            return {"error": "AI chat not initialized"}
        
        try:
            message = self.UserMessage(
                text=f"""Analyze this target URL for penetration testing: {target_url}

Please provide:
1. Technology stack detection based on URL structure
2. Suggested directories/files to test
3. Potential parameter names to fuzz
4. API endpoints that might exist
5. Security testing recommendations

Return response in JSON format."""
            )
            
            response = await self.chat.send_message(message)
            return {"analysis": response, "error": None}
            
        except Exception as e:
            return {"error": f"AI analysis failed: {str(e)}"}
    
    async def analyze_response(self, result: ScanResult) -> Dict[str, Any]:
        """Analyze HTTP response for security implications"""
        if not self.chat:
            return {"score": 0.0, "analysis": "AI not available"}
        
        try:
            message = self.UserMessage(
                text=f"""Analyze this HTTP response for security implications:

URL: {result.url}
Method: {result.method}
Status Code: {result.status_code}
Content Length: {result.content_length}
Content Type: {result.content_type}
Response Time: {result.response_time}ms
Redirect URL: {result.redirect_url}

Provide:
1. Vulnerability score (0-10)
2. Security analysis
3. Recommendations
4. Interesting findings

Keep response concise and focused on security implications."""
            )
            
            response = await self.chat.send_message(message)
            
            # Extract vulnerability score (simple heuristic)
            score = 0.0
            if result.status_code in [200, 201, 204]:
                score += 2.0
            if result.status_code in [403, 405]:
                score += 1.0
            if result.status_code == 500:
                score += 3.0
            if result.content_length > 10000:
                score += 1.0
            
            return {"score": score, "analysis": response}
            
        except Exception as e:
            return {"score": 0.0, "analysis": f"AI analysis failed: {str(e)}"}
    
    async def generate_wordlist(self, target_url: str, context: str) -> List[str]:
        """Generate custom wordlist based on target analysis"""
        if not self.chat:
            return []
        
        try:
            message = self.UserMessage(
                text=f"""Generate a custom wordlist for directory busting based on:

Target URL: {target_url}
Context: {context}

Generate 50 relevant directory/file names that might exist on this target.
Focus on common web application patterns, technology-specific paths, and security-relevant locations.
Return only the wordlist, one item per line."""
            )
            
            response = await self.chat.send_message(message)
            
            # Extract wordlist from response
            wordlist = [
                line.strip() for line in response.split('\n') 
                if line.strip() and not line.strip().startswith('#')
            ]
            
            return wordlist[:50]  # Limit to 50 items
            
        except Exception as e:
            console.print(f"[red]Failed to generate AI wordlist: {str(e)}[/red]")
            return []

class AiDirFuzz:
    """Main directory busting and fuzzing tool"""
    
    def __init__(self, config: ScanConfig):
        self.config = config
        self.results: List[ScanResult] = []
        self.session = None
        self.ai_analyzer = None
        self.wordlist_fetcher = None
        self.progress = None
        self.task_id = None
        self.total_requests = 0
        self.completed_requests = 0
        self.start_time = None
        
        # Initialize AI analyzer if API key is provided
        if config.gemini_api_key and config.ai_analysis:
            self.ai_analyzer = AIAnalyzer(config.gemini_api_key, config.gemini_model)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        global STOP_SCANNING
        STOP_SCANNING = True
        console.print("\n[red]Stopping scan... Please wait for cleanup[/red]")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_requests,
            limit_per_host=self.config.concurrent_requests,
            ttl_dns_cache=300,
            ttl_dns_cache_refresh=60
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": self.config.user_agent}
        )
        
        self.wordlist_fetcher = GitHubWordlistFetcher()
        await self.wordlist_fetcher.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        
        if self.wordlist_fetcher:
            await self.wordlist_fetcher.__aexit__(exc_type, exc_val, exc_tb)
    
    async def make_request(self, url: str, method: str = "GET", **kwargs) -> Optional[ScanResult]:
        """Make HTTP request and return result"""
        if STOP_SCANNING:
            return None
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            request_kwargs = {
                "method": method,
                "url": url,
                "allow_redirects": self.config.follow_redirects,
                "max_redirects": self.config.max_redirects,
                "ssl": self.config.verify_ssl,
                "headers": self.config.custom_headers,
                "cookies": self.config.cookies,
                **kwargs
            }
            
            # Add proxy if specified
            if self.config.proxy:
                request_kwargs["proxy"] = self.config.proxy
            
            async with self.session.request(**request_kwargs) as response:
                response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                
                # Get response info
                content_length = response.headers.get('Content-Length', 0)
                try:
                    content_length = int(content_length)
                except (ValueError, TypeError):
                    content_length = 0
                
                content_type = response.headers.get('Content-Type', 'unknown')
                redirect_url = str(response.url) if response.url != url else None
                
                result = ScanResult(
                    url=url,
                    method=method,
                    status_code=response.status,
                    content_length=content_length,
                    content_type=content_type,
                    response_time=response_time,
                    redirect_url=redirect_url
                )
                
                # AI analysis if enabled
                if self.ai_analyzer:
                    ai_result = await self.ai_analyzer.analyze_response(result)
                    result.ai_analysis = ai_result.get("analysis", "")
                    result.vulnerability_score = ai_result.get("score", 0.0)
                
                return result
                
        except asyncio.TimeoutError:
            return ScanResult(
                url=url,
                method=method,
                status_code=408,
                content_length=0,
                content_type="timeout",
                response_time=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return ScanResult(
                url=url,
                method=method,
                status_code=0,
                content_length=0,
                content_type="error",
                response_time=(time.time() - start_time) * 1000,
                ai_analysis=f"Error: {str(e)}"
            )
    
    async def directory_scan(self, wordlist: List[str]) -> List[ScanResult]:
        """Perform directory scanning"""
        results = []
        base_url = self.config.target_url.rstrip('/')
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def scan_directory(word: str):
            async with semaphore:
                if STOP_SCANNING:
                    return None
                
                # Add delay if configured
                if self.config.delay > 0:
                    await asyncio.sleep(self.config.delay)
                
                # Test directory
                dir_url = f"{base_url}/{word}"
                result = await self.make_request(dir_url)
                
                if result and self._is_interesting_result(result):
                    results.append(result)
                    self._update_progress()
                
                # Test with extensions if provided
                if self.config.extensions:
                    for ext in self.config.extensions:
                        if STOP_SCANNING:
                            break
                        
                        file_url = f"{base_url}/{word}.{ext}"
                        file_result = await self.make_request(file_url)
                        
                        if file_result and self._is_interesting_result(file_result):
                            results.append(file_result)
                            self._update_progress()
                
                self._update_progress()
                return result
        
        # Start progress tracking
        self.total_requests = len(wordlist) * (1 + len(self.config.extensions))
        self._start_progress()
        
        # Execute concurrent requests
        tasks = [scan_directory(word) for word in wordlist]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._stop_progress()
        return results
    
    async def parameter_fuzz(self, base_url: str, wordlist: List[str]) -> List[ScanResult]:
        """Perform parameter fuzzing"""
        results = []
        methods = ["GET", "POST"]
        
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def fuzz_parameter(param: str, method: str):
            async with semaphore:
                if STOP_SCANNING:
                    return None
                
                if self.config.delay > 0:
                    await asyncio.sleep(self.config.delay)
                
                if method == "GET":
                    # Test GET parameter
                    url = f"{base_url}?{param}=test"
                    result = await self.make_request(url, method="GET")
                else:
                    # Test POST parameter
                    data = {param: "test"}
                    result = await self.make_request(
                        base_url, 
                        method="POST", 
                        data=data
                    )
                
                if result and self._is_interesting_result(result):
                    results.append(result)
                
                self._update_progress()
                return result
        
        # Start progress tracking
        self.total_requests = len(wordlist) * len(methods)
        self._start_progress()
        
        # Execute concurrent requests
        tasks = []
        for param in wordlist:
            for method in methods:
                tasks.append(fuzz_parameter(param, method))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._stop_progress()
        return results
    
    async def api_discovery(self, wordlist: List[str]) -> List[ScanResult]:
        """Perform API endpoint discovery"""
        results = []
        base_url = self.config.target_url.rstrip('/')
        api_prefixes = ["/api", "/v1", "/v2", "/v3", "/rest", "/graphql"]
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def discover_endpoint(prefix: str, endpoint: str, method: str):
            async with semaphore:
                if STOP_SCANNING:
                    return None
                
                if self.config.delay > 0:
                    await asyncio.sleep(self.config.delay)
                
                url = f"{base_url}{prefix}/{endpoint}"
                result = await self.make_request(url, method=method)
                
                if result and self._is_interesting_result(result):
                    results.append(result)
                
                self._update_progress()
                return result
        
        # Start progress tracking
        self.total_requests = len(api_prefixes) * len(wordlist) * len(methods)
        self._start_progress()
        
        # Execute concurrent requests
        tasks = []
        for prefix in api_prefixes:
            for endpoint in wordlist:
                for method in methods:
                    tasks.append(discover_endpoint(prefix, endpoint, method))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self._stop_progress()
        return results
    
    def _is_interesting_result(self, result: ScanResult) -> bool:
        """Check if result is interesting based on filters"""
        # Status code filter
        if result.status_code not in self.config.status_codes:
            return False
        
        # Content length filter
        if (self.config.content_length_filter and 
            result.content_length == self.config.content_length_filter):
            return False
        
        # AI vulnerability score filter
        if result.vulnerability_score > 1.0:
            return True
        
        # Default interesting status codes
        interesting_codes = [200, 201, 204, 301, 302, 307, 308, 403, 405, 500]
        return result.status_code in interesting_codes
    
    def _start_progress(self):
        """Start progress tracking"""
        if RICH_AVAILABLE:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            )
            self.progress.start()
            self.task_id = self.progress.add_task("Scanning...", total=self.total_requests)
        
        self.start_time = time.time()
        self.completed_requests = 0
    
    def _update_progress(self):
        """Update progress"""
        self.completed_requests += 1
        if self.progress and self.task_id:
            self.progress.update(self.task_id, advance=1)
    
    def _stop_progress(self):
        """Stop progress tracking"""
        if self.progress:
            self.progress.stop()
    
    async def run_scan(self) -> List[ScanResult]:
        """Run the main scanning process"""
        console.print(f"[bold green]Starting AiDirFuzz scan against: {self.config.target_url}[/bold green]")
        console.print(f"[blue]Mode: {self.config.mode} | Concurrent: {self.config.concurrent_requests}[/blue]")
        
        # AI target analysis
        if self.ai_analyzer:
            console.print("[yellow]Running AI target analysis...[/yellow]")
            ai_analysis = await self.ai_analyzer.analyze_target(self.config.target_url)
            if ai_analysis.get("error"):
                console.print(f"[red]AI analysis failed: {ai_analysis['error']}[/red]")
            else:
                console.print("[green]AI analysis completed![/green]")
        
        # Load wordlist
        wordlist = []
        if self.config.custom_wordlist:
            # Load custom wordlist
            try:
                with open(self.config.custom_wordlist, 'r') as f:
                    wordlist = [line.strip() for line in f if line.strip()]
            except FileNotFoundError:
                console.print(f"[red]Custom wordlist file not found: {self.config.custom_wordlist}[/red]")
                return []
        else:
            # Use GitHub wordlists
            console.print("[yellow]Fetching wordlists from GitHub...[/yellow]")
            if self.config.mode == "dir":
                wordlist = await self.wordlist_fetcher.get_combined_wordlist("directories")
            elif self.config.mode == "param":
                wordlist = await self.wordlist_fetcher.get_combined_wordlist("parameters")
            elif self.config.mode == "api":
                wordlist = await self.wordlist_fetcher.get_combined_wordlist("api")
            elif self.config.mode == "hybrid":
                # Combine all wordlists
                dir_words = await self.wordlist_fetcher.get_combined_wordlist("directories")
                param_words = await self.wordlist_fetcher.get_combined_wordlist("parameters")
                api_words = await self.wordlist_fetcher.get_combined_wordlist("api")
                wordlist = list(set(dir_words + param_words + api_words))
        
        # Generate AI wordlist if enabled
        if self.ai_analyzer:
            console.print("[yellow]Generating AI-powered wordlist...[/yellow]")
            ai_words = await self.ai_analyzer.generate_wordlist(
                self.config.target_url, 
                self.config.mode
            )
            wordlist.extend(ai_words)
            wordlist = list(set(wordlist))  # Remove duplicates
        
        console.print(f"[green]Loaded {len(wordlist)} words for scanning[/green]")
        
        # Run scan based on mode
        if self.config.mode == "dir":
            results = await self.directory_scan(wordlist)
        elif self.config.mode == "param":
            results = await self.parameter_fuzz(self.config.target_url, wordlist)
        elif self.config.mode == "api":
            results = await self.api_discovery(wordlist)
        elif self.config.mode == "hybrid":
            # Run all modes
            console.print("[yellow]Running hybrid scan (all modes)...[/yellow]")
            dir_results = await self.directory_scan(wordlist)
            param_results = await self.parameter_fuzz(self.config.target_url, wordlist)
            api_results = await self.api_discovery(wordlist)
            results = dir_results + param_results + api_results
        else:
            console.print(f"[red]Unknown scan mode: {self.config.mode}[/red]")
            return []
        
        # Sort results by vulnerability score
        results.sort(key=lambda x: x.vulnerability_score, reverse=True)
        
        self.results = results
        return results
    
    def save_results(self, results: List[ScanResult]):
        """Save results to file"""
        if not self.config.output_file:
            return
        
        try:
            if self.config.output_format == "json":
                with open(self.config.output_file, 'w') as f:
                    json.dump([asdict(result) for result in results], f, indent=2)
            elif self.config.output_format == "csv":
                with open(self.config.output_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'url', 'method', 'status_code', 'content_length',
                        'content_type', 'response_time', 'vulnerability_score',
                        'ai_analysis', 'timestamp'
                    ])
                    writer.writeheader()
                    for result in results:
                        writer.writerow(asdict(result))
            elif self.config.output_format == "txt":
                with open(self.config.output_file, 'w') as f:
                    for result in results:
                        f.write(f"{result.status_code} {result.method} {result.url}\n")
            
            console.print(f"[green]Results saved to: {self.config.output_file}[/green]")
            
        except Exception as e:
            console.print(f"[red]Failed to save results: {str(e)}[/red]")
    
    def display_results(self, results: List[ScanResult]):
        """Display results in a beautiful table"""
        if not results:
            console.print("[yellow]No interesting results found[/yellow]")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="AiDirFuzz Results")
            table.add_column("URL", style="cyan")
            table.add_column("Method", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Length", style="blue")
            table.add_column("Type", style="yellow")
            table.add_column("Score", style="red")
            table.add_column("AI Analysis", style="white")
            
            for result in results[:50]:  # Show top 50 results
                score_color = "red" if result.vulnerability_score > 5 else "yellow" if result.vulnerability_score > 2 else "green"
                ai_summary = (result.ai_analysis or "")[:50] + "..." if len(result.ai_analysis or "") > 50 else (result.ai_analysis or "")
                
                table.add_row(
                    result.url,
                    result.method,
                    str(result.status_code),
                    str(result.content_length),
                    result.content_type,
                    f"[{score_color}]{result.vulnerability_score:.1f}[/{score_color}]",
                    ai_summary
                )
            
            console.print(table)
        else:
            # Fallback display
            print("\nScan Results:")
            print("-" * 80)
            for result in results[:50]:
                print(f"{result.status_code} {result.method} {result.url} [{result.vulnerability_score:.1f}]")
        
        console.print(f"\n[bold green]Total interesting results: {len(results)}[/bold green]")
        if len(results) > 50:
            console.print(f"[yellow]Showing top 50 results. Use output file to see all results.[/yellow]")

def setup_config_file():
    """Setup configuration file for API key"""
    config_dir = Path.home() / ".aifuzz"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    
    if not config_file.exists():
        # First time setup
        console.print("[bold yellow]AiDirFuzz First Time Setup[/bold yellow]")
        console.print("Please enter your Gemini API key for AI-powered analysis:")
        
        api_key = input("Gemini API Key: ").strip()
        
        config = {
            "gemini_api_key": api_key,
            "gemini_model": "gemini-2.0-flash"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print(f"[green]Configuration saved to: {config_file}[/green]")
        console.print("[yellow]You can change the API key anytime by running: aifuzz --config[/yellow]")
        
        return config
    else:
        # Load existing config
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[red]Failed to load config: {str(e)}[/red]")
            return {}

def update_config():
    """Update configuration file"""
    config_dir = Path.home() / ".aifuzz"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    
    # Load existing config
    config = {}
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            pass
    
    console.print("[bold yellow]Update AiDirFuzz Configuration[/bold yellow]")
    console.print(f"Current API Key: {config.get('gemini_api_key', 'Not set')[:10]}...")
    
    new_api_key = input("New Gemini API Key (press Enter to keep current): ").strip()
    if new_api_key:
        config["gemini_api_key"] = new_api_key
    
    print("Available Gemini models:")
    models = [
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-05-06", 
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    model_choice = input(f"Select model (1-{len(models)}, press Enter for current): ").strip()
    if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
        config["gemini_model"] = models[int(model_choice) - 1]
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"[green]Configuration updated successfully![/green]")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="AiDirFuzz - AI-Powered Directory Busting and Fuzzing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aifuzz -u https://example.com -m dir -c 100
  aifuzz -u https://api.example.com -m api -c 50 -o results.json
  aifuzz -u https://example.com -m param -w custom_params.txt
  aifuzz -u https://example.com -m hybrid -c 200 --ai-analysis
  aifuzz --config  # Update configuration
        """
    )
    
    parser.add_argument("-u", "--url", required=False, help="Target URL")
    parser.add_argument("-m", "--mode", choices=["dir", "param", "api", "hybrid"], 
                       default="dir", help="Scan mode (default: dir)")
    parser.add_argument("-c", "--concurrent", type=int, default=50, 
                       help="Concurrent requests (default: 50)")
    parser.add_argument("-t", "--timeout", type=int, default=10, 
                       help="Request timeout in seconds (default: 10)")
    parser.add_argument("-d", "--delay", type=float, default=0.0, 
                       help="Delay between requests in seconds (default: 0.0)")
    parser.add_argument("-w", "--wordlist", help="Custom wordlist file")
    parser.add_argument("-e", "--extensions", nargs="+", default=[], 
                       help="File extensions to test (e.g., php js html)")
    parser.add_argument("-s", "--status-codes", nargs="+", type=int, 
                       default=[200, 201, 204, 301, 302, 307, 308, 403, 405, 500],
                       help="Status codes to show (default: 200 201 204 301 302 307 308 403 405 500)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-f", "--format", choices=["json", "csv", "txt"], 
                       default="json", help="Output format (default: json)")
    parser.add_argument("--headers", nargs="+", help="Custom headers (format: 'Header:Value')")
    parser.add_argument("--cookies", nargs="+", help="Custom cookies (format: 'name=value')")
    parser.add_argument("--proxy", help="Proxy URL (http://proxy:port)")
    parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI analysis")
    parser.add_argument("--config", action="store_true", help="Update configuration")
    parser.add_argument("--user-agent", default="AiDirFuzz/1.0", help="Custom User-Agent")
    parser.add_argument("--version", action="version", version="AiDirFuzz 1.0.0")
    
    args = parser.parse_args()
    
    # Handle config update
    if args.config:
        update_config()
        return
    
    # Check if URL is provided
    if not args.url:
        console.print("[red]Error: Target URL is required. Use -u or --url[/red]")
        parser.print_help()
        return
    
    # Setup or load configuration
    config_data = setup_config_file()
    
    # Parse custom headers
    custom_headers = {}
    if args.headers:
        for header in args.headers:
            if ":" in header:
                key, value = header.split(":", 1)
                custom_headers[key.strip()] = value.strip()
    
    # Parse cookies
    cookies = {}
    if args.cookies:
        for cookie in args.cookies:
            if "=" in cookie:
                key, value = cookie.split("=", 1)
                cookies[key.strip()] = value.strip()
    
    # Create scan configuration
    scan_config = ScanConfig(
        target_url=args.url,
        mode=args.mode,
        concurrent_requests=args.concurrent,
        timeout=args.timeout,
        delay=args.delay,
        user_agent=args.user_agent,
        custom_headers=custom_headers,
        cookies=cookies,
        proxy=args.proxy,
        verify_ssl=not args.no_ssl_verify,
        custom_wordlist=args.wordlist,
        extensions=args.extensions,
        status_codes=args.status_codes,
        output_format=args.format,
        output_file=args.output,
        ai_analysis=not args.no_ai,
        gemini_api_key=config_data.get("gemini_api_key"),
        gemini_model=config_data.get("gemini_model", "gemini-2.0-flash")
    )
    
    # Run the scan
    async def run_scan():
        try:
            async with AiDirFuzz(scan_config) as scanner:
                results = await scanner.run_scan()
                
                # Display results
                scanner.display_results(results)
                
                # Save results if output file specified
                if args.output:
                    scanner.save_results(results)
                
                # Final summary
                elapsed_time = time.time() - scanner.start_time if scanner.start_time else 0
                console.print(f"\n[bold green]Scan completed in {elapsed_time:.2f} seconds[/bold green]")
                console.print(f"[blue]Total requests: {scanner.completed_requests}[/blue]")
                console.print(f"[blue]Requests per second: {scanner.completed_requests/elapsed_time:.2f}[/blue]")
                
        except KeyboardInterrupt:
            console.print("\n[red]Scan interrupted by user[/red]")
        except Exception as e:
            console.print(f"[red]Error during scan: {str(e)}[/red]")
    
    # Install required packages if not available
    try:
        import aiohttp
    except ImportError:
        console.print("[yellow]Installing required packages...[/yellow]")
        os.system("pip install aiohttp")
    
    try:
        from emergentintegrations.llm.chat import LlmChat
    except ImportError:
        console.print("[yellow]Installing AI integration package...[/yellow]")
        os.system("pip install emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/")
    
    # Run the scanner
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(run_scan())

if __name__ == "__main__":
    main()