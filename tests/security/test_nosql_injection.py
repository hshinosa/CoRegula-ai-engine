"""
Test NoSQL Injection - NoSQL Injection Prevention Testing
Menguji perlindungan terhadap serangan NoSQL Injection pada MongoDB
"""

import asyncio
import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class NoSQLInjectionResult:
    """Hasil pengujian NoSQL injection"""
    test_name: str
    payload: str
    injection_detected: bool
    blocked: bool
    sanitized_input: Optional[str]
    duration_ms: float
    severity: str  # HIGH, MEDIUM, LOW


class NoSQLInjectionTester:
    """Tester untuk mendeteksi dan mencegah NoSQL injection"""

    # NoSQL Injection patterns yang umum
    INJECTION_PATTERNS = {
        'operator_injection': [
            r'\$eq\s*:',
            r'\$ne\s*:',
            r'\$gt\s*:',
            r'\$gte\s*:',
            r'\$lt\s*:',
            r'\$lte\s*:',
            r'\$in\s*:',
            r'\$nin\s*:',
            r'\$regex\s*:',
            r'\$where\s*:',
            r'\$exists\s*:',
            r'\$type\s*:',
        ],
        'logical_operators': [
            r'\$and\s*:',
            r'\$or\s*:',
            r'\$not\s*:',
            r'\$nor\s*:',
        ],
        'evaluation_operators': [
            r'\$expr\s*:',
            r'\$jsonSchema\s*:',
            r'\$mod\s*:',
            r'\$text\s*:',
        ],
        'javascript_injection': [
            r'\$function\s*:',
            r'function\s*\(',
            r'\$accumulator\s*:',
            r'\$map\s*:',
        ]
    }

    def __init__(self):
        self.results: List[NoSQLInjectionResult] = []

    def detect_nosql_injection(self, user_input: str) -> tuple[bool, str, str]:
        """
        Deteksi potensi NoSQL injection dalam input

        Returns:
            Tuple of (is_injection, matched_pattern, severity)
        """
        input_lower = user_input.lower()

        # Check against all patterns
        for category, patterns in self.INJECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, input_lower, re.IGNORECASE):
                    severity = 'HIGH' if category in ['evaluation_operators', 'javascript_injection'] else 'MEDIUM'
                    return True, pattern, severity

        # Additional checks for suspicious patterns
        suspicious_patterns = [
            (r'[\{\[]\s*\$', 'Object injection attempt'),
            (r'\$\{[^}]+\}', 'Template injection'),
            (r'\btrue\b|\bfalse\b|\bnull\b', 'Boolean/Null manipulation'),
        ]

        for pattern, description in suspicious_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return True, description, 'LOW'

        return False, '', 'NONE'

    def sanitize_input(self, user_input: str) -> str:
        """
        Sanitasi input untuk mencegah NoSQL injection
        """
        # Remove MongoDB operators
        sanitized = user_input

        # Escape $ operators
        for operator in [
            '$eq', '$ne', '$gt', '$gte', '$lt', '$lte',
            '$in', '$nin', '$regex', '$where', '$exists',
            '$and', '$or', '$not', '$nor', '$expr'
        ]:
            sanitized = re.sub(
                rf'\{operator}\s*:',
                f'\\{operator}:',
                sanitized,
                flags=re.IGNORECASE
            )

        # Remove JavaScript code blocks
        sanitized = re.sub(r'function\s*\([^)]*\)\s*\{[^}]*\}', '[REMOVED]', sanitized, flags=re.IGNORECASE)

        # Escape special characters
        sanitized = sanitized.replace('$', '\\$')

        return sanitized

    def validate_query_structure(self, query: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validasi struktur query MongoDB

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for forbidden operators at top level
        forbidden_operators = ['$where', '$expr', '$function']

        def check_operators(obj, path=''):
            if isinstance(obj, dict):
                for key in obj.keys():
                    if key.startswith('$'):
                        if key in forbidden_operators:
                            return False, f"Forbidden operator '{key}' found at {path}"

                        # Check nested structures
                        result, msg = check_operators(obj[key], f"{path}.{key}" if path else key)
                        if not result:
                            return result, msg

                    elif isinstance(obj[key], (dict, list)):
                        result, msg = check_operators(obj[key], f"{path}.{key}" if path else key)
                        if not result:
                            return result, msg

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    result, msg = check_operators(item, f"{path}[{i}]")
                    if not result:
                        return result, msg

            return True, ''

        return check_operators(query)

    async def test_nosql_injection_prevention(
        self,
        test_name: str,
        payload: str
    ) -> NoSQLInjectionResult:
        """
        Test satu kasus NoSQL injection
        """
        start_time = time.time()

        # Detect injection
        is_injection, pattern, severity = self.detect_nosql_injection(payload)

        # Sanitize if injection detected
        sanitized = self.sanitize_input(payload) if is_injection else payload

        # Check if properly blocked
        blocked = is_injection and sanitized != payload

        duration = (time.time() - start_time) * 1000

        return NoSQLInjectionResult(
            test_name=test_name,
            payload=payload,
            injection_detected=is_injection,
            blocked=blocked,
            sanitized_input=sanitized if is_injection else None,
            duration_ms=duration,
            severity=severity if is_injection else 'NONE'
        )

    async def run_tests(self) -> Dict[str, Any]:
        """
        Jalankan semua test case untuk NoSQL injection
        """
        test_cases = [
            # High severity - Evaluation operators
            ('JS Code Injection', '{"username": "admin", "$where": "this.password == \'pass\'"}'),
            ('Function Injection', '{"$expr": {"$function": {"body": "return true"}}}'),
            ('Accumulation Attack', '{"$accumulator": {"init": "function() { return 0 }"}}'),

            # Medium severity - Comparison operators
            ('Operator Injection - $ne', '{"username": {"$ne": null}}'),
            ('Operator Injection - $gt', '{"age": {"$gt": ""}}'),
            ('Operator Injection - $regex', '{"username": {"$regex": "^admin"}}'),
            ('Logical OR Attack', '{"$or": [{"username": "admin"}, {"role": "admin"}]}'),

            # Low severity - Object manipulation
            ('Object Injection', '{"username": {$eq: "admin"}}'),
            ('Boolean Manipulation', '{"active": true, "role": {$ne: null}}'),

            # Safe inputs (should not be flagged)
            ('Safe String Input', 'username123'),
            ('Safe Email', 'user@example.com'),
            ('Safe Text', 'This is a normal search query'),
        ]

        print(f"\n🔒 Menjalankan {len(test_cases)} test case untuk NoSQL Injection...")

        self.results = []
        for i, (name, payload) in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] Testing: {name}...", end=' ')

            result = await self.test_nosql_injection_prevention(name, payload)
            self.results.append(result)

            if result.injection_detected:
                status = "🛡️ BLOCKED" if result.blocked else "⚠️ DETECTED"
                print(f"{status} ({result.severity})")
            else:
                print("✅ SAFE")

        # Calculate summary
        injection_attempts = [r for r in self.results if r.injection_detected]
        blocked_attempts = [r for r in self.results if r.blocked]
        high_severity = [r for r in self.results if r.severity == 'HIGH']

        summary = {
            'test_name': 'NoSQL Injection Prevention',
            'total_tests': len(self.results),
            'safe_inputs': len(self.results) - len(injection_attempts),
            'injection_detected': len(injection_attempts),
            'blocked': len(blocked_attempts),
            'block_rate': len(blocked_attempts) / len(injection_attempts) if injection_attempts else 1.0,
            'high_severity_blocked': all(r.blocked for r in high_severity) if high_severity else True,
            'avg_detection_time_ms': sum(r.duration_ms for r in self.results) / len(self.results),
            'all_high_severity_blocked': all(r.blocked for r in high_severity) if high_severity else True,
            'results': [
                {
                    'test_name': r.test_name,
                    'payload': r.payload[:50] + '...' if len(r.payload) > 50 else r.payload,
                    'injection_detected': r.injection_detected,
                    'blocked': r.blocked,
                    'severity': r.severity,
                    'duration_ms': r.duration_ms
                }
                for r in self.results
            ]
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print summary hasil test"""
        print("\n" + "="*60)
        print("🛡️ NOSQL INJECTION PREVENTION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests:        {summary['total_tests']}")
        print(f"Safe Inputs:        {summary['safe_inputs']} ✅")
        print(f"Injections Found:   {summary['injection_detected']} ⚠️")
        print(f"Blocked:            {summary['blocked']} 🛡️")
        print(f"Block Rate:         {summary['block_rate']:.1%}")
        print(f"Avg Detection Time: {summary['avg_detection_time_ms']:.1f}ms")
        print()

        # Print high severity results
        high_severity_results = [r for r in self.results if r.severity == 'HIGH']
        if high_severity_results:
            print("High Severity Attacks:")
            for r in high_severity_results:
                status = "🛡️ BLOCKED" if r.blocked else "❌ NOT BLOCKED"
                print(f"  • {r.test_name}: {status}")
            print()

        status = "✅ ALL TESTS PASSED" if summary['all_high_severity_blocked'] else "❌ SOME TESTS FAILED"
        print(f"Status: {status}")
        print("="*60)


async def main():
    """Main function untuk menjalankan NoSQL injection tests"""
    print("🛡️ NoSQL Injection Prevention Testing")
    print("Menguji perlindungan terhadap serangan NoSQL injection\n")

    tester = NoSQLInjectionTester()

    # Run tests
    summary = await tester.run_tests()
    tester.print_summary(summary)

    # Save results
    output_file = 'nosql_injection_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return summary


if __name__ == '__main__':
    asyncio.run(main())
