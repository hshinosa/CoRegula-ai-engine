"""
Verify Load Test Setup
======================

Script untuk verifikasi setup load testing.
Usage: python verify_setup.py

Issue: KOL-42
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"[OK] Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  [WARN] Python 3.8+ recommended")
        return False
    return True


def check_locust():
    """Check if locust is installed."""
    try:
        result = subprocess.run(
            ["locust", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[OK] Locust installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("[FAIL] Locust not installed")
    print("  Run: pip install locust")
    return False


def check_locustfile():
    """Check locustfile syntax."""
    locustfile = Path(__file__).parent / "locustfile.py"
    
    if not locustfile.exists():
        print(f"[FAIL] locustfile.py not found at {locustfile}")
        return False
    
    try:
        # Try to compile the file
        with open(locustfile, 'r', encoding='utf-8') as f:
            compile(f.read(), locustfile, 'exec')
        print("[OK] locustfile.py syntax valid")
        return True
    except SyntaxError as e:
        print(f"[FAIL] locustfile.py has syntax error: {e}")
        return False


def check_generate_report():
    """Check generate_report.py syntax."""
    report_script = Path(__file__).parent / "generate_report.py"
    
    if not report_script.exists():
        print(f"[FAIL] generate_report.py not found")
        return False
    
    try:
        with open(report_script, 'r', encoding='utf-8') as f:
            compile(f.read(), report_script, 'exec')
        print("[OK] generate_report.py syntax valid")
        return True
    except SyntaxError as e:
        print(f"[FAIL] generate_report.py has syntax error: {e}")
        return False


def check_run_load_test():
    """Check run_load_test.py syntax."""
    runner_script = Path(__file__).parent / "run_load_test.py"
    
    if not runner_script.exists():
        print(f"[FAIL] run_load_test.py not found")
        return False
    
    try:
        with open(runner_script, 'r', encoding='utf-8') as f:
            compile(f.read(), runner_script, 'exec')
        print("[OK] run_load_test.py syntax valid")
        return True
    except SyntaxError as e:
        print(f"[FAIL] run_load_test.py has syntax error: {e}")
        return False


def check_directory_structure():
    """Check directory structure."""
    base = Path(__file__).parent
    
    required_files = [
        "locustfile.py",
        "run_load_test.py",
        "generate_report.py",
        "requirements-loadtest.txt",
        "README.md",
    ]
    
    all_exist = True
    for file in required_files:
        path = base / file
        if path.exists():
            print(f"[OK] {file} exists")
        else:
            print(f"[FAIL] {file} missing")
            all_exist = False
    
    # Check results directory
    results_dir = base / "results"
    if results_dir.exists():
        print("[OK] results/ directory exists")
    else:
        print("[WARN] results/ directory not found (will be created on first run)")
    
    return all_exist


def check_ai_engine_connection(host="http://localhost:8001"):
    """Check if AI Engine is reachable."""
    import urllib.request
    import urllib.error
    
    try:
        req = urllib.request.Request(
            f"{host}/health",
            method='GET',
            headers={'Accept': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print(f"[OK] AI Engine responding at {host}")
                return True
    except urllib.error.URLError as e:
        print(f"[FAIL] Cannot connect to AI Engine at {host}")
        print(f"  Error: {e}")
        print(f"  Make sure AI Engine is running: python -m uvicorn app.main:app --port 8001")
        return False
    except Exception as e:
        print(f"[WARN] Connection check failed: {e}")
        return False


def main():
    print("="*60)
    print("Load Testing Setup Verification")
    print("="*60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Locust Installation", check_locust),
        ("Directory Structure", check_directory_structure),
        ("Locustfile Syntax", check_locustfile),
        ("Generate Report Script", check_generate_report),
        ("Run Load Test Script", check_run_load_test),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        results.append((name, check_func()))
    
    print(f"\nAI Engine Connection:")
    results.append(("AI Engine Connection", check_ai_engine_connection()))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print("="*60)
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\nAll checks passed! Ready to run load tests.")
        print("\nNext steps:")
        print("  1. python run_load_test.py --scenario smoke")
        print("  2. python run_load_test.py --scenario load")
        return 0
    else:
        print("\nSome checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
