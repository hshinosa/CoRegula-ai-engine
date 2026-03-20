#!/bin/bash
# Security Audit Script for CoRegula AI Engine
# KOL-144: Container & Dependency Hardening

set -e

echo "🔒 Running security audit..."

# Create reports directory
mkdir -p reports

# Audit Python dependencies
echo "📦 Auditing Python dependencies with pip-audit..."
if command -v pip-audit &> /dev/null; then
    pip-audit -r requirements.txt -f json -o reports/dependency-audit.json || true
    echo "✅ Dependency audit complete: reports/dependency-audit.json"
    
    # Check for high/critical vulnerabilities
    echo "🔍 Checking for high/critical vulnerabilities..."
    if pip-audit -r requirements.txt --fail-threshold high; then
        echo "✅ No high/critical vulnerabilities found"
    else
        echo "❌ High/critical vulnerabilities found! Check reports/dependency-audit.json"
        exit 1
    fi
else
    echo "⚠️  pip-audit not installed. Install with: pip install pip-audit"
fi

# Run bandit for code security
echo "🔍 Running bandit security scan..."
if command -v bandit &> /dev/null; then
    bandit -r app/ -f json -o reports/code-audit.json || true
    echo "✅ Code security scan complete: reports/code-audit.json"
else
    echo "⚠️  bandit not installed. Install with: pip install bandit"
fi

# Check for hardcoded secrets
echo "🔐 Checking for hardcoded secrets..."
if grep -r "sk-" --include="*.py" app/ | grep -v ".pyc" | grep -v "__pycache__"; then
    echo "❌ Potential hardcoded API keys found!"
    echo "   Remove all hardcoded secrets and use environment variables instead."
    exit 1
else
    echo "✅ No hardcoded secrets detected"
fi

# Check for HTTP URLs in production config
echo "🔒 Checking for insecure URLs..."
if grep -r "http://[^l]" --include="*.py" app/core/config.py | grep -v "localhost" | grep -v "127.0.0.1"; then
    echo "⚠️  Found non-localhost HTTP URLs - ensure these use HTTPS in production"
fi

echo ""
echo "✅ Security audit complete!"
echo "📊 Reports saved to: reports/"
