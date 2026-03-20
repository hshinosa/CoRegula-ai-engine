"""
Load Test Report Generator
==========================

Generate HTML report dari hasil load test.
Usage: python generate_report.py --csv-prefix results/ai_engine_load_test --output report.html

Issue: KOL-42
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import csv


def parse_stats_csv(filepath):
    """Parse Locust stats CSV file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in ['Request Count', 'Failure Count', 'Median Response Time', 
                           'Average Response Time', 'Min Response Time', 'Max Response Time',
                           'Average Content Size', 'Requests/s', 'Failures/s']:
                    if key in row:
                        try:
                            row[key] = float(row[key])
                        except:
                            row[key] = 0
                data.append(row)
    except FileNotFoundError:
        return []
    return data


def parse_history_csv(filepath):
    """Parse Locust request history CSV file."""
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        return []
    return data


def generate_html_report(stats_data, history_data, failures_data, output_path):
    """Generate HTML report dari data load test."""
    
    # Calculate summary metrics
    total_requests = sum(row.get('Request Count', 0) for row in stats_data)
    total_failures = sum(row.get('Failure Count', 0) for row in stats_data)
    failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0
    
    # Find slowest endpoints
    sorted_by_avg = sorted(stats_data, key=lambda x: x.get('Average Response Time', 0), reverse=True)
    slowest_endpoints = sorted_by_avg[:5]
    
    # Find highest throughput
    sorted_by_rps = sorted(stats_data, key=lambda x: x.get('Requests/s', 0), reverse=True)
    highest_throughput = sorted_by_rps[:5]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Engine Load Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.success {{ color: #10b981; }}
        .metric-value.warning {{ color: #f59e0b; }}
        .metric-value.error {{ color: #ef4444; }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e5e5e5;
        }}
        tr:hover {{
            background: #f9fafb;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .status-pass {{
            background: #d1fae5;
            color: #065f46;
        }}
        .status-fail {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .status-warn {{
            background: #fef3c7;
            color: #92400e;
        }}
        .recommendations {{
            background: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 20px;
            margin-top: 20px;
        }}
        .recommendations h3 {{
            margin: 0 0 15px 0;
            color: #065f46;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 10px;
            color: #374151;
        }}
        .timestamp {{
            color: #9ca3af;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 AI Engine Load Test Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Issue: KOL-42 | Load Testing Performa Response Time AI</p>
    </div>
    
    <div class="summary-grid">
        <div class="metric-card">
            <h3>Total Requests</h3>
            <div class="metric-value">{int(total_requests):,}</div>
        </div>
        <div class="metric-card">
            <h3>Failed Requests</h3>
            <div class="metric-value {'error' if failure_rate > 5 else 'success'}">{int(total_failures):,}</div>
        </div>
        <div class="metric-card">
            <h3>Failure Rate</h3>
            <div class="metric-value {'error' if failure_rate > 5 else 'success'}">{failure_rate:.2f}%</div>
        </div>
        <div class="metric-card">
            <h3>Status</h3>
            <div class="metric-value {'success' if failure_rate < 1 else 'warning' if failure_rate < 5 else 'error'}">
                {'PASS' if failure_rate < 1 else 'WARN' if failure_rate < 5 else 'FAIL'}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Endpoint Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Method</th>
                    <th>Requests</th>
                    <th>Failures</th>
                    <th>Avg Response (ms)</th>
                    <th>Median (ms)</th>
                    <th>Min (ms)</th>
                    <th>Max (ms)</th>
                    <th>Req/s</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for row in sorted(stats_data, key=lambda x: x.get('Request Count', 0), reverse=True):
        name = row.get('Name', 'Unknown')
        method = row.get('Method', 'GET')
        requests = int(row.get('Request Count', 0))
        failures = int(row.get('Failure Count', 0))
        avg_time = row.get('Average Response Time', 0)
        median = row.get('Median Response Time', 0)
        min_time = row.get('Min Response Time', 0)
        max_time = row.get('Max Response Time', 0)
        rps = row.get('Requests/s', 0)
        
        failure_pct = (failures / requests * 100) if requests > 0 else 0
        if failure_pct < 1 and avg_time < 1000:
            status = '<span class="status-badge status-pass">PASS</span>'
        elif failure_pct < 5 and avg_time < 3000:
            status = '<span class="status-badge status-warn">WARN</span>'
        else:
            status = '<span class="status-badge status-fail">FAIL</span>'
        
        html += f"""
                <tr>
                    <td><code>{name}</code></td>
                    <td>{method}</td>
                    <td>{requests:,}</td>
                    <td>{failures:,}</td>
                    <td>{avg_time:.1f}</td>
                    <td>{median:.1f}</td>
                    <td>{min_time:.1f}</td>
                    <td>{max_time:.1f}</td>
                    <td>{rps:.2f}</td>
                    <td>{status}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>⏱️ Slowest Endpoints</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Endpoint</th>
                    <th>Avg Response Time</th>
                    <th>Max Response Time</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, row in enumerate(slowest_endpoints, 1):
        name = row.get('Name', 'Unknown')
        avg_time = row.get('Average Response Time', 0)
        max_time = row.get('Max Response Time', 0)
        
        if avg_time > 5000:
            rec = "🚨 Critical: Consider async processing or caching"
        elif avg_time > 2000:
            rec = "⚠️ Slow: Review query optimization"
        else:
            rec = "✅ Acceptable"
        
        html += f"""
                <tr>
                    <td>{i}</td>
                    <td><code>{name}</code></td>
                    <td>{avg_time:.1f} ms</td>
                    <td>{max_time:.1f} ms</td>
                    <td>{rec}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>⚡ Highest Throughput Endpoints</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Endpoint</th>
                    <th>Requests/s</th>
                    <th>Total Requests</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for i, row in enumerate(highest_throughput, 1):
        name = row.get('Name', 'Unknown')
        rps = row.get('Requests/s', 0)
        requests = int(row.get('Request Count', 0))
        
        html += f"""
                <tr>
                    <td>{i}</td>
                    <td><code>{name}</code></td>
                    <td>{rps:.2f}</td>
                    <td>{requests:,}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>❌ Failed Requests</h2>
"""
    
    if failures_data:
        html += """
        <table>
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Endpoint</th>
                    <th>Error</th>
                    <th>Occurrences</th>
                </tr>
            </thead>
            <tbody>
"""
        for row in failures_data:
            html += f"""
                <tr>
                    <td>{row.get('Method', 'N/A')}</td>
                    <td><code>{row.get('Name', 'Unknown')}</code></td>
                    <td>{row.get('Error', 'Unknown')}</td>
                    <td>{row.get('Occurrences', 0)}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""
    else:
        html += '<p style="color: #10b981; font-size: 1.2em;">✅ No failures recorded!</p>'
    
    html += """
    </div>
    
    <div class="section">
        <div class="recommendations">
            <h3>📝 Recommendations</h3>
            <ul>
"""
    
    # Generate recommendations based on data
    recommendations = []
    
    if failure_rate > 5:
        recommendations.append("<strong>Critical:</strong> Failure rate terlalu tinggi (>5%). Periksa error logs dan capacity server.")
    
    if slowest_endpoints and slowest_endpoints[0].get('Average Response Time', 0) > 5000:
        recommendations.append(f"<strong>Performance:</strong> Endpoint {slowest_endpoints[0].get('Name')} lambat (>5s). Pertimbangkan caching atau optimasi query.")
    
    recommendations.append("Monitor cache hit rate di endpoint /efficiency/cache/statistics.")
    recommendations.append("Pertimbangkan rate limiting jika traffic meningkat signifikan.")
    recommendations.append("Review query RAG untuk optimasi embedding retrieval.")
    
    for rec in recommendations:
        html += f"                <li>{rec}</li>\n"
    
    html += """            </ul>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 SLA Targets</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Target</th>
                    <th>Actual</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Response Time (p95)</td>
                    <td>&lt; 3000ms</td>
                    <td>See Max column</td>
                    <td>Review</td>
                </tr>
                <tr>
                    <td>Response Time (avg)</td>
                    <td>&lt; 1000ms</td>
                    <td>See Avg column</td>
                    <td>Review</td>
                </tr>
                <tr>
                    <td>Error Rate</td>
                    <td>&lt; 1%</td>
                    <td>{failure_rate:.2f}%</td>
                    <td>{'✅ PASS' if failure_rate < 1 else '❌ FAIL'}</td>
                </tr>
                <tr>
                    <td>Availability</td>
                    <td>&gt; 99%</td>
                    <td>{100-failure_rate:.2f}%</td>
                    <td>{'✅ PASS' if failure_rate < 1 else '⚠️ REVIEW'}</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <footer style="text-align: center; color: #9ca3af; padding: 20px;">
        <p>AI Engine Load Testing Suite | CoRegula Project</p>
    </footer>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate load test report')
    parser.add_argument('--csv-prefix', required=True, help='Prefix for CSV files')
    parser.add_argument('--output', default='load_test_report.html', help='Output HTML file')
    
    args = parser.parse_args()
    
    # Read CSV files
    stats_file = f"{args.csv_prefix}_stats.csv"
    history_file = f"{args.csv_prefix}_stats_history.csv"
    failures_file = f"{args.csv_prefix}_failures.csv"
    
    print(f"Reading stats from: {stats_file}")
    stats_data = parse_stats_csv(stats_file)
    
    print(f"Reading history from: {history_file}")
    history_data = parse_history_csv(history_file)
    
    print(f"Reading failures from: {failures_file}")
    failures_data = parse_stats_csv(failures_file)  # Same format as stats
    
    if not stats_data:
        print("❌ No stats data found!")
        sys.exit(1)
    
    print(f"✅ Loaded {len(stats_data)} endpoints, {len(failures_data)} failures")
    
    generate_html_report(stats_data, history_data, failures_data, args.output)


if __name__ == '__main__':
    main()
