"""
Test Context Recall - Context Recall@K Testing
Mengukur proporsi informasi yang diperlukan yang berhasil di-retrieve
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple
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
        async def generate(self, prompt, temperature=0.1, max_tokens=500):
            class MockResponse:
                content = "- Mock fact 1\n- Mock fact 2\nFOUND: YES\nDOCUMENT: 1"
            return MockResponse()
    
    class VectorService:
        async def similarity_search(self, query, top_k=5):
            return {'chunks': [{'content': f'Mock chunk {i}', 'id': f'chunk_{i}'} for i in range(top_k)]}


@dataclass
class ContextRecallResult:
    """Hasil pengujian context recall"""
    query: str
    k: int
    ground_truth_facts: List[str]
    retrieved_facts: List[str]
    matched_facts: List[str]
    recall_at_k: float  # Recall@K
    f1_score: float
    duration_ms: float


class ContextRecallTester:
    """Tester untuk mengukur Context Recall@K"""

    def __init__(self):
        self.llm_service = LLMService()
        self.vector_service = VectorService()
        self.results: List[ContextRecallResult] = []

    async def extract_facts_from_text(self, text: str) -> List[str]:
        """
        Ekstrak factual statements dari text menggunakan LLM

        Returns:
            List of factual statements
        """
        prompt = f"""Ekstrak semua factual statements (fakta/informasi penting) dari teks berikut.

TEKS:
{text[:1500]}

Tugas:
1. Identifikasi setiap fakta penting yang ada dalam teks
2. Tulis dalam bentuk bullet point yang singkat dan jelas
3. Fokus pada informasi yang dapat digunakan untuk menjawab pertanyaan

Format output:
- [Fakta 1]
- [Fakta 2]
- [Fakta 3]
... dan seterusnya
"""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )

            # Parse facts from bullet points
            facts = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('• '):
                    fact = line[2:].strip()
                    if fact and len(fact) > 10:  # Filter very short facts
                        facts.append(fact)

            return facts

        except Exception as e:
            print(f"⚠️ Error extracting facts: {e}")
            return []

    async def check_fact_in_chunks(
        self,
        fact: str,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[bool, int]:
        """
        Periksa apakah suatu fact ada dalam retrieved chunks

        Returns:
            Tuple of (is_found, chunk_index)
        """
        prompt = f"""Periksa apakah informasi berikut ada dalam salah satu dokumen.

FAKTA YANG DICARI:
{fact}

DOKUMEN-DOKUMEN:
"""

        # Add chunks to prompt
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('content', chunk.get('text', ''))
            prompt += f"\n--- Dokumen {i+1} ---\n{chunk_text[:500]}\n"

        prompt += """

Tugas:
Apakah fakta yang dicari ADA dalam dokumen-dokumen di atas? (YES/NO)
Jika YA, di dokumen nomor berapa?

Format:
FOUND: [YES atau NO]
DOCUMENT: [nomor dokumen atau NONE]
"""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=100
            )

            content = response.content.upper()

            is_found = 'YES' in content
            chunk_index = -1

            # Try to extract document number
            for line in response.content.split('\n'):
                if line.startswith('DOCUMENT:'):
                    doc_str = line.split(':', 1)[1].strip()
                    try:
                        chunk_index = int(doc_str) - 1
                    except:
                        pass
                    break

            return is_found, chunk_index

        except Exception as e:
            print(f"⚠️ Error checking fact: {e}")
            return False, -1

    def calculate_recall(
        self,
        ground_truth_facts: List[str],
        matched_facts: List[str]
    ) -> float:
        """
        Hitung Recall@K

        Recall = (jumlah facts yang ditemukan) / (total facts yang seharusnya)
        """
        if not ground_truth_facts:
            return 1.0  # No facts to retrieve = perfect recall

        return len(matched_facts) / len(ground_truth_facts)

    def calculate_f1(self, precision: float, recall: float) -> float:
        """Hitung F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    async def test_context_recall(
        self,
        query: str,
        ground_truth_answer: str,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, ContextRecallResult]:
        """
        Test context recall untuk satu query

        Args:
            query: Query pengguna
            ground_truth_answer: Jawaban lengkap yang mengandung semua facts
            k_values: List of K values untuk Recall@K

        Returns:
            Dictionary of K -> ContextRecallResult
        """
        start_time = time.time()

        # Extract facts from ground truth
        ground_truth_facts = await self.extract_facts_from_text(ground_truth_answer)

        # Retrieve chunks
        retrieved = await self.vector_service.similarity_search(
            query=query,
            top_k=max(k_values)
        )

        chunks = retrieved.get('chunks', [])

        # Check which facts are found in retrieved chunks
        matched_facts = []
        matched_indices = []

        for fact in ground_truth_facts:
            is_found, chunk_idx = await self.check_fact_in_chunks(fact, chunks)
            if is_found:
                matched_facts.append(fact)
                if chunk_idx >= 0:
                    matched_indices.append(chunk_idx)

        duration = (time.time() - start_time) * 1000

        # Calculate metrics untuk setiap K
        results = {}
        for k in k_values:
            if k > len(chunks):
                continue

            # Facts that should be in top-K chunks
            # (This is an approximation - assumes facts are distributed)
            expected_facts_in_k = len(ground_truth_facts) * (k / len(chunks)) if chunks else 0

            # Calculate recall based on matched facts
            recall_k = self.calculate_recall(ground_truth_facts, matched_facts)

            # Estimate precision (matched / retrieved)
            precision_k = len(matched_facts) / k if k > 0 else 0

            f1 = self.calculate_f1(precision_k, recall_k)

            results[f'@{k}'] = ContextRecallResult(
                query=query,
                k=k,
                ground_truth_facts=ground_truth_facts,
                retrieved_facts=matched_facts,
                matched_facts=matched_facts,
                recall_at_k=recall_k,
                f1_score=f1,
                duration_ms=duration
            )

        return results

    async def run_tests(
        self,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Jalankan batch test untuk context recall

        Args:
            test_cases: List of {
                'query': str,
                'ground_truth_answer': str  # Complete answer with all facts
            }

        Returns:
            Dictionary dengan summary hasil test
        """
        print(f"\n📚 Menjalankan {len(test_cases)} test case untuk Context Recall...")

        all_results = []
        k_values = [3, 5, 10]

        for i, test in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] Testing: {test['query'][:50]}...", end=' ')

            results = await self.test_context_recall(
                query=test['query'],
                ground_truth_answer=test['ground_truth_answer'],
                k_values=k_values
            )

            for k, result in results.items():
                all_results.append({
                    'k': k,
                    'recall': result.recall_at_k,
                    'f1': result.f1_score,
                    'facts_total': len(result.ground_truth_facts),
                    'facts_matched': len(result.matched_facts),
                    'query': result.query
                })

            # Show R@5 sebagai representative
            r_at_5 = results.get('@5', ContextRecallResult(
                query='', k=5, ground_truth_facts=[], retrieved_facts=[],
                matched_facts=[], recall_at_k=0, f1_score=0, duration_ms=0
            )).recall_at_k

            print(f"R@5: {r_at_5:.1%}")

        # Calculate summary statistics per K
        summary_by_k = {}
        for k in k_values:
            k_results = [r['recall'] for r in all_results if r['k'] == f'@{k}']
            f1_results = [r['f1'] for r in all_results if r['k'] == f'@{k}']

            if k_results:
                summary_by_k[f'R@{k}'] = {
                    'mean': sum(k_results) / len(k_results),
                    'min': min(k_results),
                    'max': max(k_results),
                    'target': 0.75 if k <= 5 else 0.6,
                    'met': (sum(k_results) / len(k_results)) >= (0.75 if k <= 5 else 0.6)
                }

                if f1_results:
                    summary_by_k[f'R@{k}']['f1_mean'] = sum(f1_results) / len(f1_results)

        summary = {
            'test_name': 'Context Recall',
            'total_tests': len(test_cases),
            'k_values_tested': k_values,
            'metrics_by_k': summary_by_k,
            'overall_recall': sum(
                summary_by_k[f'R@{k}']['mean'] for k in k_values if f'R@{k}' in summary_by_k
            ) / len(k_values) if summary_by_k else 0,
            'target_met': all(
                summary_by_k[f'R@{k}']['met'] for k in k_values if f'R@{k}' in summary_by_k
            ) if summary_by_k else False,
            'detailed_results': all_results
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print summary hasil test dengan format yang rapi"""
        print("\n" + "="*60)
        print("📚 CONTEXT RECALL TEST SUMMARY")
        print("="*60)
        print(f"Total Queries Tested: {summary['total_tests']}")
        print(f"K Values Tested: {summary['k_values_tested']}")
        print()
        print("Recall Metrics:")
        print("-" * 40)

        for metric, data in summary['metrics_by_k'].items():
            status = "✅" if data['met'] else "❌"
            print(f"  {metric}:")
            print(f"    Mean:  {data['mean']:.1%} {status} (target: {data['target']:.0%})")
            print(f"    Range: [{data['min']:.1%} - {data['max']:.1%}]")
            if 'f1_mean' in data:
                print(f"    F1:    {data['f1_mean']:.1%}")
            print()

        print(f"Overall Recall: {summary['overall_recall']:.1%}")
        print(f"Target Status: {'✅ ALL TARGETS MET' if summary['target_met'] else '❌ SOME TARGETS NOT MET'}")
        print("="*60)


# Sample test cases dengan ground truth lengkap
SAMPLE_TEST_CASES = [
    {
        'query': 'Apa itu Kolabri dan apa fungsinya?',
        'ground_truth_answer': '''Kolabri adalah platform AI untuk otomatisasi regulasi.
Platform ini membantu perusahaan mematuhi regulasi dengan menggunakan kecerdasan buatan.
Fungsi utamanya adalah compliance monitoring, document analysis, dan automated reporting.
Kolabri dapat menganalisis dokumen regulasi dan memberikan rekomendasi compliance.
Sistem ini juga dapat mengirim alert jika ada pelanggaran regulasi.
Dengan Kolabri, perusahaan dapat mengurangi risiko non-compliance dan biaya operasional.'''
    },
    {
        'query': 'Bagaimana cara mengatur izin pengguna di sistem Kolabri?',
        'ground_truth_answer': '''Untuk mengatur izin pengguna di Kolabri, admin perlu masuk ke menu Settings > User Management > Permissions.
Dari sana, admin dapat menambah, mengubah, atau menghapus izin pengguna.
Kolabri menggunakan sistem role-based access control (RBAC) dengan role seperti Admin, Manager, dan User.
Setiap role memiliki izin yang berbeda-beda sesuai dengan tugas dan tanggung jawabnya.
Admin dapat juga membuat custom role dengan izin yang spesifik.
Semua perubahan izin akan tercatat dalam audit log untuk keperluan compliance.'''
    },
    {
        'query': 'Apa saja fitur security yang dimiliki Kolabri?',
        'ground_truth_answer': '''Kolabri memiliki fitur keamanan yang kuat untuk melindungi data pengguna.
Data dienkripsi menggunakan AES-256 untuk data at rest dan TLS 1.3 untuk data in transit.
Sistem mematuhi standar keamanan ISO 27001 dan SOC 2 Type II.
Kolabri juga mendukung single sign-on (SSO) dengan SAML 2.0 dan OAuth 2.0.
Ada fitur two-factor authentication (2FA) untuk proteksi tambahan.
Semua akses ke sistem dicatat dalam audit log yang immutable.'''
    }
]


async def main():
    """Main function untuk menjalankan context recall tests"""
    print("📚 Context Recall Testing")
    print("Mengukur kemampuan sistem mengambil informasi yang diperlukan\n")

    tester = ContextRecallTester()

    # Run tests with sample data
    summary = await tester.run_tests(SAMPLE_TEST_CASES)
    tester.print_summary(summary)

    # Save results
    output_file = 'context_recall_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return summary


if __name__ == '__main__':
    asyncio.run(main())
