"""
High Performance Load Test Runner
==================================

Script untuk menjalankan load test sesuai target performance:
- Baseline: 10 users, 50 RPS, <200ms P95
- Beban Sedang: 50 users, 250 RPS, <300ms P95
- Beban Puncak: 100 users, 500 RPS, <500ms P95
- Uji Stres: 500 users, 2500 RPS, <100ms P95

Usage:
    python run_high_perf_test.py --scenario baseline
    python run_high_perf_test.py --scenario beban_sedang
    python run_high_perf_test.py --scenario beban_puncak
    python run_high_perf_test.py --scenario uji_stres

Issue: KOL-42 - High Performance Targets
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime
from pathlib import Path


# Target configurations
TARGETS = {
    "baseline": {
        "description": "Baseline: 10 users, 50 RPS, <200ms P95",
        "users": 10,
        "spawn_rate": 5,
        "duration": "5m",
        "target_rps": 50,
        "target_p95_ms": 200,
        "user_class": "BaselineUser",
    },
    "beban_sedang": {
        "description": "Beban Sedang: 50 users, 250 RPS, <300ms P95",
        "users": 50,
        "spawn_rate": 10,
        "duration": "10m",
        "target_rps": 250,
        "target_p95_ms": 300,
        "user_class": "BebanSedangUser",
    },
    "beban_puncak": {
        "description": "Beban Puncak: 100 users, 500 RPS, <500ms P95",
        "users": 100,
        "spawn_rate": 20,
        "duration": "15m",
        "target_rps": 500,
        "target_p95_ms": 500,
        "user_class": "BebanPuncakUser",
    },
    "uji_stres": {
        "description": "Uji Stres: 500 users, 2500 RPS, <100ms P95",
        "users": 500,
        "spawn_rate": 100,
        "duration": "5m",
        "target_rps": 2500,
        "target_p95_ms": 100,
        "user_class": "UjiStresUser",
    },
    "spike": {
        "description": "Spike Test: Sudden traffic surge",
        "users": 100,
        "spawn_rate": 50,
        "duration": "3m",
        "target_rps": 500,
        "target_p95_ms": 1000,
        "user_class": "SpikeTestUser",
    },
    "endurance": {
        "description": "Endurance Test: 30 users, 150 RPS, 30 minutes",
        "users": 30,
        "spawn_rate": 5,
        "duration": "30m",
        "target_rps": 150,
        "target_p95_ms": 500,
        "user_class": "EnduranceUser",
    },
}


def check_locust_installed():
    """Check if locust is installed."""
    try:
        subprocess.run(["locust", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_load_test(scenario_config, host, output_dir, target_name):
    """Run locust load test."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_prefix = f"{output_dir}/{target_name}_{timestamp}"
    
    cmd = [
        "locust",
        "-f", "locustfile_high_perf.py",
        "--host", host,
        "--users", str(scenario_config["users"]),
        "--spawn-rate", str(scenario_config["spawn_rate"]),
        "--run-time", scenario_config["duration"],
        "--headless",
        "--csv", csv_prefix,
    ]
    
    print(f"\n{'='*80}")
    print(f"🚀 STARTING: {scenario_config['description']}")
    print(f"{'='*80}")
    print(f"Users: {scenario_config['users']}")
    print(f"Spawn Rate: {scenario_config['spawn_rate']}/s")
    print(f"Duration: {scenario_config['duration']}")
    print(f"Target RPS: {scenario_config['target_rps']}")
    print(f"Target P95: {scenario_config['target_p95_ms']}ms")
    print(f"Host: {host}")
    print(f"Output: {csv_prefix}")
    print(f"{'='*80}\n")
    
    # Pre-warm cache untuk uji stres
    if target_name == "uji_stres":
        print("⚡ Pre-warming cache untuk uji stres...")
        print("   Pastikan endpoint /ask/batch/precomputed sudah dipopulate.")
        print("   Run: python prewarm_cache.py\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Test completed: {scenario_config['description']}")
        return csv_prefix
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error code: {e.returncode}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run High Performance Load Tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Skenario Performance Test:
  baseline      : 10 users, 50 RPS, <200ms P95, 5 menit
  beban_sedang  : 50 users, 250 RPS, <300ms P95, 10 menit
  beban_puncak  : 100 users, 500 RPS, <500ms P95, 15 menit
  uji_stres     : 500 users, 2500 RPS, <100ms P95, 5 menit
  spike         : 100 users, spike test, 3 menit
  endurance     : 30 users, 150 RPS, 30 menit endurance

Contoh:
  python run_high_perf_test.py --scenario baseline
  python run_high_perf_test.py --scenario uji_stres --host http://prod.kolabri.ai
        """
    )
    
    parser.add_argument(
        '--scenario',
        choices=list(TARGETS.keys()),
        default='baseline',
        help='Skenario test (default: baseline)'
    )
    parser.add_argument(
        '--host',
        default='http://localhost:8001',
        help='Target host (default: http://localhost:8001)'
    )
    parser.add_argument(
        '--output',
        default='./results',
        help='Output directory (default: ./results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check locust
    if not check_locust_installed():
        print("❌ Locust not installed!")
        print("💡 Run: pip install locust")
        sys.exit(1)
    
    # Get scenario config
    scenario_config = TARGETS[args.scenario]
    
    # Run test
    result = run_load_test(scenario_config, args.host, str(output_dir), args.scenario)
    
    if result:
        print(f"\n{'='*80}")
        print("📊 LOAD TEST COMPLETED")
        print(f"{'='*80}")
        print(f"Scenario: {scenario_config['description']}")
        print(f"CSV files: {result}_*.csv")
        print(f"\nLihat hasil di:")
        print(f"  - CSV: {result}_stats.csv")
        print(f"  - History: {result}_stats_history.csv")
        print(f"  - Failures: {result}_failures.csv")
        print(f"\nGenerate report:")
        print(f"  python generate_report.py --csv-prefix {result} --output report_{args.scenario}.html")
        print(f"{'='*80}")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
