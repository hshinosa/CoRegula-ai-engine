"""
Test Context Precision - Context Precision@K Testing
Mengukur proporsi chunk yang relevan dari total chunk yang di-retrieve
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.services.llm.llm_service import LLMService
    from src.services.vector.vector_service import VectorService
except ImportError:
    # Mock for standalone testing
    class LLMService:
        async def generate(self, prompt, temperature=0.1, max_tokens=100):
            class MockResponse:
                content = "RELEVANT: YES\nREASON: Mock"
            return MockResponse()
    
    class VectorService:
        async def similarity_search(self, query, top_k=5):
            return {'chunks': [{'content': f'Mock chunk {i}', 'id': f'chunk_{i}'} for i in range(top_k)]}


@dataclass
class ContextPrecisionResult:
    """Hasil pengujian context precision"""
    query: str
    k: int  # Top-K retrieved
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[int]  # Indices of relevant chunks
    precision_at_k: float  # Precision@K
    avg_precision: float  # Average Precision (AP)
    duration_ms: float


class ContextPrecisionTester:
    """Tester untuk mengukur Context Precision@K"""

    def __init__(self):
        self.llm_service = LLMService()
        self.vector_service = VectorService()
        self.results: List[ContextPrecisionResult] = []

    async def is_chunk_relevant(
        self,
        query: str,
        chunk: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Evaluasi apakah chunk relevan dengan query menggunakan LLM

        Returns:
            Tuple of (is_relevant, reasoning)
        """
        chunk_text = chunk.get('content', chunk.get('text', ''))

        judge_prompt = f"""Anda adalah evaluator yang menilai relevansi dokumen terhadap pertanyaan.

PERTANYAAN:
{query}

DOKUMEN:
{chunk_text[:1000]}

Tugas Anda:
1. Nilai apakah dokumen di atas RELEVAN untuk menjawab pertanyaan (YES/NO)
2. Berikan penjelasan singkat (1 kalimat)

Kriteria relevansi:
- Dokumen mengandung informasi yang LANGSUNG membantu menjawab pertanyaan
- Dokumen berhubungan dengan topik yang ditanyakan

Format jawaban:
RELEVANT: [YES atau NO]
REASON: [penjelasan singkat]
"""

        try:
            response = await self.llm_service.generate(
                prompt=judge_prompt,
                temperature=0.1,
                max_tokens=100
            )

            content = response.content.upper()

            # Parse result
            is_relevant = 'YES' in content and 'NO' not in content.split('YES')[0]

            # Extract reasoning
            reasoning = "No reasoning provided"
            for line in response.content.split('\n'):
                if line.startswith('REASON:'):
                    reasoning = line.split(':', 1)[1].strip()
                    break

            return is_relevant, reasoning

        except Exception as e:
            print(f"⚠️ Error judging relevance: {e}")
            return False, f"Error: {str(e)}"

    def calculate_precision_at_k(
        self,
        relevance_judgments: List[bool],
        k: int
    ) -> float:
        """
        Hitung Precision@K

        Precision@K = (jumlah dokumen relevan di top-K) / K
        """
        if k == 0 or len(relevance_judgments) == 0:
            return 0.0

        # Take only top-K
        top_k = relevance_judgments[:k]

        # Count relevant documents
        relevant_count = sum(1 for r in top_k if r)

        return relevant_count / k

    def calculate_average_precision(
        self,
        relevance_judgments: List[bool]
    ) -> float:
        """
        Hitung Average Precision (AP)

        AP = average of precision@k for each k where doc k is relevant
        """
        if not any(relevance_judgments):
            return 0.0

        precisions = []
        relevant_count = 0

        for i, is_relevant in enumerate(relevance_judgments, 1):
            if is_relevant:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precisions.append(precision_at_i)

        if not precisions:
            return 0.0

        return sum(precisions) / len(precisions)

    async def test_context_precision(
        self,
        query: str,
        expected_chunks: List[str],
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, ContextPrecisionResult]:
        """
        Test context precision untuk satu query

        Args:
            query: Query pengguna
            expected_chunks: List of expected/relevant chunk contents (ground truth)
            k_values: List of K values untuk Precision@K

        Returns:
            Dictionary of K -> ContextPrecisionResult
        """
        start_time = time.time()

        # Retrieve chunks using vector search
        retrieved = await self.vector_service.similarity_search(
            query=query,
            top_k=max(k_values)
        )

        chunks = retrieved.get('chunks', [])

        # Evaluate relevance of each chunk
        relevance_judgments = []
        for i, chunk in enumerate(chunks):
            is_rel, reason = await self.is_chunk_relevant(query, chunk)
            relevance_judgments.append(is_rel)

        duration = (time.time() - start_time) * 1000

        # Calculate metrics untuk setiap K
        results = {}
        for k in k_values:
            if k > len(chunks):
                continue

            relevant_indices = [
                i for i, is_rel in enumerate(relevance_judgments[:k]) if is_rel
            ]

            precision_k = self.calculate_precision_at_k(relevance_judgments, k)
            avg_precision = self.calculate_average_precision(relevance_judgments[:k])

            results[f'@{k}'] = ContextPrecisionResult(
                query=query,
                k=k,
                retrieved_chunks=[
                    {
                        'id': chunk.get('id', f'chunk_{i}'),
                        'content': chunk.get('content', '')[:200] + '...',
                        'score': chunk.get('score', 0)
                    }
                    for i, chunk in enumerate(chunks[:k])
                ],
                relevant_chunks=relevant_indices,
                precision_at_k=precision_k,
                avg_precision=avg_precision,
                duration_ms=duration
            )

        return results

    async def run_tests(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Jalankan batch test untuk context precision

        Args:
            test_cases: List of {
                'query': str,
                'expected_chunks': List[str],
                'expected_precision': Optional[float]
            }

        Returns:
            Dictionary dengan summary hasil test
        """
        print(f"\n🎯 Menjalankan {len(test_cases)} test case untuk Context Precision...")

        all_results = []
        k_values = [3, 5, 10]

        for i, test in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] Testing: {test['query'][:50]}...", end=' ')

            results = await self.test_context_precision(
                query=test['query'],
                expected_chunks=test.get('expected_chunks', []),
                k_values=k_values
            )

            for k, result in results.items():
                all_results.append({
                    'k': k,
                    'precision': result.precision_at_k,
                    'query': result.query
                })

            # Show P@5 sebagai representative
            p_at_5 = results.get('@5', ContextPrecisionResult(
                query='', k=5, retrieved_chunks=[], relevant_chunks=[],
                precision_at_k=0, avg_precision=0, duration_ms=0
            )).precision_at_k

            print(f"P@5: {p_at_5:.1%}")

        # Calculate summary statistics per K
        summary_by_k = {}
        for k in k_values:
            k_results = [r['precision'] for r in all_results if r['k'] == f'@{k}']
            if k_results:
                summary_by_k[f'P@{k}'] = {
                    'mean': sum(k_results) / len(k_results),
                    'min': min(k_results),
                    'max': max(k_results),
                    'target': 0.8 if k <= 5 else 0.6,
                    'met': (sum(k_results) / len(k_results)) >= (0.8 if k <= 5 else 0.6)
                }

        summary = {
            'test_name': 'Context Precision',
            'total_tests': len(test_cases),
            'k_values_tested': k_values,
            'metrics_by_k': summary_by_k,
            'overall_precision': sum(
                summary_by_k[f'P@{k}']['mean'] for k in k_values if f'P@{k}' in summary_by_k
            ) / len(k_values) if summary_by_k else 0,
            'target_met': all(
                summary_by_k[f'P@{k}']['met'] for k in k_values if f'P@{k}' in summary_by_k
            ) if summary_by_k else False,
            'detailed_results': all_results
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print summary hasil test dengan format yang rapi"""
        print("\n" + "="*60)
        print("🎯 CONTEXT PRECISION TEST SUMMARY")
        print("="*60)
        print(f"Total Queries Tested: {summary['total_tests']}")
        print(f"K Values Tested: {summary['k_values_tested']}")
        print()
        print("Precision Metrics:")
        print("-" * 40)

        for metric, data in summary['metrics_by_k'].items():
            status = "✅" if data['met'] else "❌"
            print(f"  {metric}:")
            print(f"    Mean:  {data['mean']:.1%} {status} (target: {data['target']:.0%})")
            print(f"    Range: [{data['min']:.1%} - {data['max']:.1%}]")
            print()

        print(f"Overall Precision: {summary['overall_precision']:.1%}")
        print(f"Target Status: {'✅ ALL TARGETS MET' if summary['target_met'] else '❌ SOME TARGETS NOT MET'}")
        print("="*60)


# Sample test cases dengan ground truth
SAMPLE_TEST_CASES = [
    {
        'query': 'Apa itu Kolabri?',
        'expected_chunks': [
            'Kolabri adalah platform AI untuk otomatisasi regulasi',
            'Platform ini membantu perusahaan mematuhi regulasi dengan AI'
        ]
    },
    {
        'query': 'Bagaimana cara mengatur izin pengguna?',
        'expected_chunks': [
            'User permissions dapat diatur melalui menu Settings',
            'Admin dapat mengelola role dan akses pengguna'
        ]
    },
    {
        'query': 'Fitur compliance monitoring',
        'expected_chunks': [
            'Compliance monitoring memantau status kepatuhan perusahaan',
            'Sistem akan mengirim alert jika ada pelanggaran regulasi'
        ]
    }
]


async def main():
    """Main function untuk menjalankan context precision tests"""
    print("🎯 Context Precision Testing")
    print("Mengukur proporsi chunk relevan dalam hasil retrieval\n")

    tester = ContextPrecisionTester()

    # Run tests with sample data
    summary = await tester.run_tests(SAMPLE_TEST_CASES)
    tester.print_summary(summary)

    # Save results
    output_file = 'context_precision_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return summary


if __name__ == '__main__':
    asyncio.run(main())
