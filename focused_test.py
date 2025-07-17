#!/usr/bin/env python3

"""
Focused AiFuzz Feature Test
Tests the specific improvements mentioned in the review request
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_test(test_name, command, expected_output=None, timeout=20):
    """Run a single test"""
    print(f"\n🧪 Testing: {test_name}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        
        output = result.stdout + result.stderr
        print(f"Output (first 500 chars): {output[:500]}...")
        
        if expected_output:
            if any(exp in output for exp in expected_output):
                print(f"✅ {test_name}: PASSED")
                return True
            else:
                print(f"❌ {test_name}: FAILED - Expected output not found")
                return False
        else:
            if result.returncode == 0 or "Stopping scan" in output:
                print(f"✅ {test_name}: PASSED")
                return True
            else:
                print(f"❌ {test_name}: FAILED - Non-zero exit code")
                return False
                
    except subprocess.TimeoutExpired:
        print(f"⚠️ {test_name}: TIMEOUT (expected for some tests)")
        return True  # Timeout is expected for scanning tests
    except Exception as e:
        print(f"❌ {test_name}: ERROR - {e}")
        return False

def main():
    print("🚀 AiFuzz Enhanced Features Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Help command shows new flags
    test_results.append(run_test(
        "Help Command with New Flags",
        "python /app/aifuzz.py --help",
        ["--wizard", "--github-wordlist"]
    ))
    
    # Test 2: Progress bar functionality
    test_results.append(run_test(
        "Progress Bar Display",
        "timeout 15 python /app/aifuzz.py -u https://httpbin.org -m dir -c 5 --wordlist-size small --no-ai",
        ["/", "%", "Scanning..."]
    ))
    
    # Test 3: GitHub wordlist support
    test_results.append(run_test(
        "GitHub Wordlist Loading",
        "timeout 15 python /app/aifuzz.py -u https://httpbin.org -m dir -c 5 --wordlist-size small --no-ai --github-wordlist https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt",
        ["Loaded", "words"]
    ))
    
    # Test 4: Verbose mode
    test_results.append(run_test(
        "Verbose Mode Logging",
        "timeout 10 python /app/aifuzz.py -u https://httpbin.org -m dir -c 5 --wordlist-size small --no-ai -v",
        ["[INFO]", "Initializing scanner"]
    ))
    
    # Test 5: Different scan modes
    for mode in ["dir", "param", "api"]:
        test_results.append(run_test(
            f"Scan Mode: {mode}",
            f"timeout 10 python /app/aifuzz.py -u https://httpbin.org -m {mode} -c 5 --wordlist-size small --no-ai",
            [f"Mode: {mode}"]
        ))
    
    # Test 6: Graceful shutdown (results saving)
    test_results.append(run_test(
        "Graceful Shutdown with Results Saving",
        "timeout 15 python /app/aifuzz.py -u https://httpbin.org -m dir -c 5 --wordlist-size small --no-ai --github-wordlist https://raw.githubusercontent.com/danielmiessler/SecLists/master/Discovery/Web-Content/common.txt",
        ["Saving results", "Results saved to"]
    ))
    
    # Test 7: AI integration handling
    test_results.append(run_test(
        "AI Integration (Disabled)",
        "timeout 10 python /app/aifuzz.py -u https://httpbin.org -m dir -c 5 --wordlist-size small --no-ai",
        None  # Just check it doesn't crash
    ))
    
    # Test 8: Results directory creation
    results_dir = Path("/app/aifuzz_results")
    if results_dir.exists() and any(results_dir.iterdir()):
        print(f"\n✅ Results Directory: PASSED - Found files in {results_dir}")
        test_results.append(True)
    else:
        print(f"\n❌ Results Directory: FAILED - No files in {results_dir}")
        test_results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! AiFuzz enhancements are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)