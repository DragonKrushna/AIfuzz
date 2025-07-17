#!/bin/bash

echo "=== AiDirFuzz v1.1.0 - Enhanced with Auto-Save Results ==="
echo ""

echo "1. Show tool version and help (with new auto-save feature):"
echo "   python aifuzz.py --version"
python aifuzz.py --version
echo ""

echo "2. Show help with auto-save information:"
echo "   python aifuzz.py --help | grep -A15 'Results are automatically'"
python aifuzz.py --help | grep -A15 'Results are automatically'
echo ""

echo "3. Test with actual results generation and auto-save:"
echo "   Create test wordlist with httpbin.org endpoints..."
echo "status" > /tmp/test_wordlist.txt
echo "json" >> /tmp/test_wordlist.txt
echo "get" >> /tmp/test_wordlist.txt
echo "post" >> /tmp/test_wordlist.txt
echo "put" >> /tmp/test_wordlist.txt

echo "   Running: python aifuzz.py -u https://httpbin.org -m dir -c 2 -w /tmp/test_wordlist.txt --no-ai -s 200 404 -f json"
python aifuzz.py -u https://httpbin.org -m dir -c 2 -w /tmp/test_wordlist.txt --no-ai -s 200 404 -f json
echo ""

echo "4. Check saved results:"
echo "   ls -la aifuzz_results/"
ls -la aifuzz_results/
echo ""

echo "5. Show JSON result format:"
echo "   Latest JSON file content:"
latest_json=$(ls -t aifuzz_results/*.json | head -1)
if [[ -f "$latest_json" ]]; then
    echo "   File: $latest_json"
    cat "$latest_json" | head -20
    echo "   ... (truncated)"
fi
echo ""

echo "6. Test TXT format:"
echo "   Running: python aifuzz.py -u https://httpbin.org -m dir -c 2 -w /tmp/test_wordlist.txt --no-ai -s 200 404 -f txt"
python aifuzz.py -u https://httpbin.org -m dir -c 2 -w /tmp/test_wordlist.txt --no-ai -s 200 404 -f txt
echo ""

echo "7. Show TXT result format:"
echo "   Latest TXT file content:"
latest_txt=$(ls -t aifuzz_results/*.txt | head -1)
if [[ -f "$latest_txt" ]]; then
    echo "   File: $latest_txt"
    cat "$latest_txt"
fi
echo ""

echo "=== NEW AUTO-SAVE FUNCTIONALITY ==="
echo ""
echo "✅ Automatic Results Saving:"
echo "   - Results are automatically saved to 'aifuzz_results/' folder"
echo "   - No need to specify -o parameter (but still supported)"
echo "   - Automatic filename generation with format: domain_mode_timestamp.format"
echo "   - Examples: httpbin.org_dir_20250717_103025.json"
echo ""
echo "✅ Enhanced Result Files:"
echo "   - JSON format includes comprehensive scan metadata"
echo "   - TXT format includes scan information header"
echo "   - CSV format maintains compatibility with existing tools"
echo ""
echo "✅ Folder Structure:"
echo "   - All results organized in aifuzz_results/ directory"
echo "   - Automatic directory creation if it doesn't exist"
echo "   - Timestamped filenames prevent overwrites"
echo ""
echo "✅ Result Metadata (JSON format):"
echo "   - Target URL and scan mode"
echo "   - Scan date and duration"
echo "   - Wordlist size and configuration"
echo "   - Total requests and interesting results count"
echo "   - AI analysis status and tool version"
echo ""
echo "=== USAGE EXAMPLES ==="
echo ""
echo "# Basic scan with auto-save:"
echo "aifuzz -u https://example.com -m dir --wordlist-size small"
echo ""
echo "# Scan with custom output file (still uses aifuzz_results/ folder):"
echo "aifuzz -u https://example.com -m dir -o custom_scan.json"
echo ""
echo "# Different formats:"
echo "aifuzz -u https://example.com -m api -f csv"
echo "aifuzz -u https://example.com -m hybrid -f txt"
echo ""
echo "# Results are automatically saved to:"
echo "# ./aifuzz_results/domain_mode_timestamp.format"
echo ""
echo "All previous improvements PLUS automatic structured result saving!"