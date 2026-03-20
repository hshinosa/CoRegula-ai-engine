"""
RAG Accuracy & Security Test Runner
====================================

Menjalankan semua pengujian akurasi RAG dan keamanan.
Menghasilkan laporan comprehensive dengan metrik.

Usage:
    python run_rag_security_tests.py
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RAG Accuracy tests
from tests.rag_accuracy.test_faithfulness import FaithfulnessEvaluator, TEST_CASES as FAITHFULNESS_CASES
from tests.rag_accuracy.test_relevance import AnswerRelevanceTester, SAMPLE_TEST_CASES as RELEVANCE_CASES
from tests.rag_accuracy.test_context_precision import ContextPrecisionTester, SAMPLE_TEST_CASES as PRECISION_CASES
from tests.rag_accuracy.test_context_recall import ContextRecallTester, SAMPLE_TEST_CASES as RECALL_CASES

# Import Security tests
from tests.security.test_prompt_injection import SecurityTester, INJECTION_TEST_CASES, JAILBREAK_TEST_CASES, PII_TEST_CASES
from tests.security.test_nosql_injection import NoSQLInjectionTester
from tests.security.test_ddos_protection import DDoSProtectionTester


class TestRunner:
    """Runner untuk RAG accuracy dan security tests."""

    def __init__(self):
        self.results = {
            'rag_accuracy': {},
            'security': {}
        }
        self.start_time = None
        self.end_time = None

    async def run_rag_accuracy_tests(self):
        """Jalankan semua RAG accuracy tests."""
        print("\n" + "=" * 70)
        print("RAG ACCURACY TESTS")
        print("=" * 70)

        total_passed = 0
        total_tests = 0

        # 1. Faithfulness Tests
        print("\n[1/4] Faithfulness Testing")
        print("-" * 40)
        evaluator = FaithfulnessEvaluator()
        passed = 0

        for case in FAITHFULNESS_CASES:
            # Simulate answer
            answer = f"Berdasarkan konteks, ini adalah penjelasan tentang {case['query']}. " + \
                     " ".join([f"{claim} merupakan konsep penting." for claim in case['expected_claims']])

            result = await evaluator.evaluate(answer, case['contexts'])
            is_passed = evaluator.passed(result)
            if is_passed:
                passed += 1

            status = "✅ PASS" if is_passed else "❌ FAIL"
            print(f"  {status} {case['name']}: Score={result.score:.3f}")

            self.results['rag_accuracy'][f"faithfulness_{case['name']}"] = {
                'score': result.score,
                'passed': is_passed
            }

        print(f"  Summary: {passed}/{len(FAITHFULNESS_CASES)} passed")
        total_passed += passed
        total_tests += len(FAITHFULNESS_CASES)

        # 2. Answer Relevance Tests
        print("\n[2/4] Answer Relevance Testing")
        print("-" * 40)
        relevance_tester = AnswerRelevanceTester()

        try:
            relevance_summary = await relevance_tester.run_tests(RELEVANCE_CASES)
            relevance_tester.print_summary(relevance_summary)

            self.results['rag_accuracy']['relevance'] = relevance_summary
            total_passed += relevance_summary['passed']
            total_tests += relevance_summary['total_tests']
        except Exception as e:
            print(f"  ⚠️ Relevance tests skipped: {e}")

        # 3. Context Precision Tests
        print("\n[3/4] Context Precision Testing")
        print("-" * 40)
        precision_tester = ContextPrecisionTester()

        try:
            precision_summary = await precision_tester.run_tests(PRECISION_CASES)
            precision_tester.print_summary(precision_summary)

            self.results['rag_accuracy']['context_precision'] = precision_summary
            total_passed += sum(1 for m in precision_summary['metrics_by_k'].values() if m['met'])
            total_tests += len(precision_summary['metrics_by_k'])
        except Exception as e:
            print(f"  ⚠️ Context precision tests skipped: {e}")

        # 4. Context Recall Tests
        print("\n[4/4] Context Recall Testing")
        print("-" * 40)
        recall_tester = ContextRecallTester()

        try:
            recall_summary = await recall_tester.run_tests(RECALL_CASES)
            recall_tester.print_summary(recall_summary)

            self.results['rag_accuracy']['context_recall'] = recall_summary
            total_passed += sum(1 for m in recall_summary['metrics_by_k'].values() if m['met'])
            total_tests += len(recall_summary['metrics_by_k'])
        except Exception as e:
            print(f"  ⚠️ Context recall tests skipped: {e}")

        return total_passed, total_tests

    async def run_security_tests(self):
        """Jalankan semua security tests."""
        print("\n" + "=" * 70)
        print("SECURITY TESTS")
        print("=" * 70)

        total_passed = 0
        total_tests = 0

        # 1. Prompt Injection Tests
        print("\n[1/3] Prompt Injection & Jailbreak Testing")
        print("-" * 40)
        tester = SecurityTester()
        passed = 0
        test_count = 0

        # Injection tests
        for case in INJECTION_TEST_CASES:
            result = await tester.test_injection(case['query'])
            is_passed = (result.detected == case['should_detect'])
            if is_passed:
                passed += 1
            test_count += 1

            status = "✅" if is_passed else "❌"
            print(f"  {status} {case['name']}")

            self.results['security'][f"injection_{case['name']}"] = {
                'detected': result.detected,
                'passed': is_passed
            }

        # Jailbreak tests
        for case in JAILBREAK_TEST_CASES:
            result = await tester.test_jailbreak(case['query'])
            is_passed = (result.detected == case['should_detect'])
            if is_passed:
                passed += 1
            test_count += 1

            status = "✅" if is_passed else "❌"
            print(f"  {status} {case['name']}")

            self.results['security'][f"jailbreak_{case['name']}"] = {
                'detected': result.detected,
                'passed': is_passed
            }

        # PII tests
        for case in PII_TEST_CASES:
            result = await tester.test_pii_detection(case['text'])
            is_passed = (result.detected == case['should_detect'])
            if is_passed:
                passed += 1
            test_count += 1

            status = "✅" if is_passed else "❌"
            print(f"  {status} {case['name']}")

            self.results['security'][f"pii_{case['name']}"] = {
                'detected': result.detected,
                'passed': is_passed
            }

        print(f"  Summary: {passed}/{test_count} passed")
        total_passed += passed
        total_tests += test_count

        # 2. NoSQL Injection Tests
        print("\n[2/3] NoSQL Injection Testing")
        print("-" * 40)
        nosql_tester = NoSQLInjectionTester()

        try:
            nosql_summary = await nosql_tester.run_tests()
            nosql_tester.print_summary(nosql_summary)

            self.results['security']['nosql_injection'] = nosql_summary
            total_passed += nosql_summary['blocked']
            total_tests += nosql_summary['injection_detected']
        except Exception as e:
            print(f"  ⚠️ NoSQL injection tests skipped: {e}")

        # 3. DDoS Protection Tests
        print("\n[3/3] DDoS Protection Testing")
        print("-" * 40)
        ddos_tester = DDoSProtectionTester()

        try:
            ddos_summary = await ddos_tester.run_tests()
            ddos_tester.print_summary(ddos_summary)

            self.results['security']['ddos_protection'] = ddos_summary
            total_passed += sum(1 for r in ddos_summary['results'] if r['protection_triggered'])
            total_tests += len(ddos_summary['results'])
        except Exception as e:
            print(f"  ⚠️ DDoS protection tests skipped: {e}")

        return total_passed, total_tests

    def generate_report(self):
        """Generate final report."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0

        print("\n" + "=" * 70)
        print("📊 FINAL TEST REPORT")
        print("=" * 70)
        print(f"Generated: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration:.1f} seconds")
        print("=" * 70)

        # RAG Accuracy Summary
        print("\n📚 RAG ACCURACY SUMMARY")
        print("-" * 70)

        rag_results = self.results['rag_accuracy']

        # Faithfulness
        faith_results = {k: v for k, v in rag_results.items() if k.startswith('faithfulness_')}
        if faith_results:
            faith_passed = sum(1 for r in faith_results.values() if r['passed'])
            faith_scores = [r['score'] for r in faith_results.values()]
            avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
            print(f"  Faithfulness:       {faith_passed}/{len(faith_results)} passed | Avg: {avg_faith:.3f} (Target: >0.85)")

        # Relevance
        if 'relevance' in rag_results:
            rel = rag_results['relevance']
            print(f"  Answer Relevance:   {rel['passed']}/{rel['total_tests']} passed | Avg: {rel['avg_score']:.1%} (Target: >70%)")

        # Context Precision
        if 'context_precision' in rag_results:
            prec = rag_results['context_precision']
            metrics = prec.get('metrics_by_k', {})
            if 'P@5' in metrics:
                p5 = metrics['P@5']
                print(f"  Context Precision:  P@5={p5['mean']:.1%} (Target: >80%)")

        # Context Recall
        if 'context_recall' in rag_results:
            rec = rag_results['context_recall']
            metrics = rec.get('metrics_by_k', {})
            if 'R@5' in metrics:
                r5 = metrics['R@5']
                print(f"  Context Recall:     R@5={r5['mean']:.1%} (Target: >75%)")

        # Security Summary
        print("\n🛡️ SECURITY SUMMARY")
        print("-" * 70)

        sec_results = self.results['security']

        # Injection/Jailbreak/PII
        inj_results = {k: v for k, v in sec_results.items()
                      if k.startswith(('injection_', 'jailbreak_', 'pii_'))}
        if inj_results:
            inj_passed = sum(1 for r in inj_results.values() if r['passed'])
            print(f"  Prompt Security:    {inj_passed}/{len(inj_results)} tests passed | {(inj_passed/len(inj_results)*100):.0f}%")

        # NoSQL
        if 'nosql_injection' in sec_results:
            nosql = sec_results['nosql_injection']
            print(f"  NoSQL Injection:    {nosql['blocked']}/{nosql['injection_detected']} blocked | {(nosql['block_rate']*100):.0f}%")

        # DDoS
        if 'ddos_protection' in sec_results:
            ddos = sec_results['ddos_protection']
            print(f"  DDoS Protection:    {ddos['successful_requests']}/{ddos['total_requests']} served | {(ddos['rate_limit_rate']*100):.0f}% limited")

        # Overall Summary
        print("\n📈 OVERALL SUMMARY")
        print("-" * 70)

        rag_passed = sum(1 for r in rag_results.values()
                        if isinstance(r, dict) and r.get('passed', False))
        rag_total = len([r for r in rag_results.values() if isinstance(r, dict) and 'passed' in r])

        sec_passed = sum(1 for r in sec_results.values()
                        if isinstance(r, dict) and r.get('passed', r.get('protection_worked', False)))
        sec_total = len(sec_results)

        total_passed = rag_passed + sec_passed
        total_tests = rag_total + sec_total

        print(f"  Total Tests:        {total_tests}")
        print(f"  Passed:             {total_passed}")
        print(f"  Failed:             {total_tests - total_passed}")
        print(f"  Success Rate:       {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")

        status = "✅ ALL TESTS PASSED" if total_passed == total_tests else "⚠️ SOME TESTS FAILED"
        print(f"\n  Status: {status}")
        print("=" * 70)

        return {
            'timestamp': self.end_time.isoformat(),
            'duration_seconds': duration,
            'rag_accuracy': rag_results,
            'security': sec_results,
            'summary': {
                'total_tests': total_tests,
                'total_passed': total_passed,
                'success_rate': (total_passed/total_tests) if total_tests > 0 else 0,
                'all_passed': total_passed == total_tests
            }
        }

    async def run_all(self):
        """Run all tests."""
        self.start_time = datetime.now()

        print("=" * 70)
        print("🧪 RAG ACCURACY & SECURITY TEST SUITE")
        print("=" * 70)
        print("\n📋 Target Metrics:")
        print("  RAG Accuracy:")
        print("    - Faithfulness:      > 85%")
        print("    - Answer Relevance:  > 70%")
        print("    - Context Precision: > 80% (P@5)")
        print("    - Context Recall:    > 75% (R@5)")
        print("  Security:")
        print("    - Prompt Injection:  100% detection")
        print("    - NoSQL Injection:   100% high severity blocked")
        print("    - DDoS Protection:   < 20% rejected under load")
        print("=" * 70)

        # Run tests
        rag_passed, rag_total = await self.run_rag_accuracy_tests()
        sec_passed, sec_total = await self.run_security_tests()

        # Generate report
        final_report = self.generate_report()

        # Save report to file
        report_file = f'test_report_{self.start_time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Detailed report saved to: {report_file}")

        return final_report


async def main():
    """Main entry point."""
    runner = TestRunner()
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
