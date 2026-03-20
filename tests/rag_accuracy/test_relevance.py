"""
Test Relevance - Answer Relevance Testing
Mengukur sejauh mana jawaban relevan dengan pertanyaan yang diajukan
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.services.llm.llm_service import LLMService
    from src.core.utils.metrics import MetricsCollector
except ImportError:
    # Mock for standalone testing
    class LLMService:
        async def get_embedding(self, text):
            class MockResponse:
                embedding = [0.1] * 1536
            return MockResponse()
        async def generate(self, prompt, temperature=0.1, max_tokens=150):
            class MockResponse:
                content = "SCORE: 75\nREASONING: Mock response"
            return MockResponse()
    
    class MetricsCollector:
        pass


@dataclass
class RelevanceResult:
    """Hasil pengujian relevansi"""
    query: str
    answer: str
    relevance_score: float  # 0-1
    semantic_similarity: float  # 0-1
    keyword_overlap: float  # 0-1
    is_relevant: bool
    reasoning: str
    duration_ms: float


class AnswerRelevanceTester:
    """Tester untuk mengukur relevansi jawaban terhadap pertanyaan"""

    def __init__(self):
        self.llm_service = LLMService()
        self.metrics = MetricsCollector()
        self.results: List[RelevanceResult] = []

    async def calculate_semantic_similarity(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Hitung semantic similarity menggunakan embeddings

        Returns:
            Score 0-1, semakin tinggi semakin mirip secara semantik
        """
        try:
            # Get embeddings for both texts
            query_embedding_response = await self.llm_service.get_embedding(query)
            answer_embedding_response = await self.llm_service.get_embedding(answer)

            # Extract embedding vectors
            query_embedding = query_embedding_response.embedding
            answer_embedding = answer_embedding_response.embedding

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, answer_embedding)

            return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to 0-1
        except Exception as e:
            print(f"⚠️ Error calculating semantic similarity: {e}")
            return 0.5  # Default neutral score

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def calculate_keyword_overlap(
        self,
        query: str,
        answer: str
    ) -> float:
        """
        Hitung keyword overlap antara query dan answer

        Returns:
            Score 0-1 berdasarkan proporsi keyword yang muncul
        """
        import re

        # Extract important keywords (nouns, verbs, numbers)
        def extract_keywords(text: str) -> set:
            # Remove punctuation and convert to lowercase
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            words = text.split()

            # Filter: keep words with length > 3 and not common stop words
            stop_words = {
                'yang', 'dan', 'dari', 'dalam', 'untuk', 'dengan', 'pada',
                'the', 'and', 'for', 'with', 'from', 'this', 'that', 'are',
                'is', 'was', 'were', 'been', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should'
            }

            keywords = {
                word for word in words
                if len(word) > 3 and word not in stop_words
            }
            return keywords

        query_keywords = extract_keywords(query)
        answer_keywords = extract_keywords(answer)

        if not query_keywords:
            return 1.0  # No keywords to match

        # Calculate overlap
        overlap = query_keywords & answer_keywords
        overlap_score = len(overlap) / len(query_keywords)

        return overlap_score

    async def llm_judge_relevance(
        self,
        query: str,
        answer: str
    ) -> tuple[float, str]:
        """
        Gunakan LLM sebagai judge untuk menilai relevansi

        Returns:
            Tuple of (score 0-1, reasoning)
        """
        judge_prompt = f"""Anda adalah evaluator yang menilai relevansi jawaban terhadap pertanyaan.

PERTANYAAN:
{query}

JAWABAN:
{answer}

Tugas Anda:
1. Nilai sejauh mana jawaban tersebut RELEVAN dengan pertanyaan (0-100)
2. Berikan penjelasan singkat (1-2 kalimat)

Format jawaban:
SCORE: [angka 0-100]
REASONING: [penjelasan singkat]
"""

        try:
            response = await self.llm_service.generate(
                prompt=judge_prompt,
                temperature=0.1,
                max_tokens=150
            )

            content = response.content

            # Parse score
            score = 0.5  # Default
            reasoning = "Unable to parse LLM response"

            for line in content.split('\n'):
                if line.startswith('SCORE:'):
                    try:
                        score = int(line.split(':')[1].strip()) / 100.0
                    except:
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()

            return max(0.0, min(1.0, score)), reasoning

        except Exception as e:
            print(f"⚠️ LLM judge error: {e}")
            return 0.5, f"Error: {str(e)}"

    async def test_answer_relevance(
        self,
        query: str,
        answer: str
    ) -> RelevanceResult:
        """
        Test relevansi jawaban terhadap pertanyaan

        Returns:
            RelevanceResult dengan composite score
        """
        start_time = time.time()

        # Calculate individual scores
        semantic_sim = await self.calculate_semantic_similarity(query, answer)
        keyword_overlap = self.calculate_keyword_overlap(query, answer)
        llm_score, reasoning = await self.llm_judge_relevance(query, answer)

        # Composite score (weighted average)
        # LLM judge: 50%, Semantic: 30%, Keywords: 20%
        composite_score = (
            llm_score * 0.5 +
            semantic_sim * 0.3 +
            keyword_overlap * 0.2
        )

        # Threshold for relevance
        is_relevant = composite_score >= 0.7

        duration = (time.time() - start_time) * 1000

        return RelevanceResult(
            query=query,
            answer=answer,
            relevance_score=composite_score,
            semantic_similarity=semantic_sim,
            keyword_overlap=keyword_overlap,
            is_relevant=is_relevant,
            reasoning=reasoning,
            duration_ms=duration
        )

    async def run_tests(
        self,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Jalankan batch test untuk relevansi

        Args:
            test_cases: List of {'query': str, 'answer': str}

        Returns:
            Dictionary dengan summary hasil test
        """
        print(f"\n🧪 Menjalankan {len(test_cases)} test case untuk Answer Relevance...")

        self.results = []
        total_duration = 0

        for i, test in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] Testing: {test['query'][:50]}...", end=' ')

            result = await self.test_answer_relevance(
                query=test['query'],
                answer=test['answer']
            )

            self.results.append(result)
            total_duration += result.duration_ms

            status = "✅" if result.is_relevant else "❌"
            print(f"{status} Score: {result.relevance_score:.2%}")

        # Calculate summary statistics
        scores = [r.relevance_score for r in self.results]
        relevant_count = sum(1 for r in self.results if r.is_relevant)

        summary = {
            'test_name': 'Answer Relevance',
            'total_tests': len(self.results),
            'passed': relevant_count,
            'failed': len(self.results) - relevant_count,
            'pass_rate': relevant_count / len(self.results) if self.results else 0,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'avg_duration_ms': total_duration / len(self.results) if self.results else 0,
            'target_threshold': 0.7,
            'target_met': (sum(scores) / len(scores)) >= 0.7 if scores else False,
            'results': [
                {
                    'query': r.query,
                    'answer': r.answer[:100] + '...' if len(r.answer) > 100 else r.answer,
                    'score': r.relevance_score,
                    'semantic_similarity': r.semantic_similarity,
                    'keyword_overlap': r.keyword_overlap,
                    'is_relevant': r.is_relevant,
                    'duration_ms': r.duration_ms
                }
                for r in self.results
            ]
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print summary hasil test dengan format yang rapi"""
        print("\n" + "="*60)
        print("📊 ANSWER RELEVANCE TEST SUMMARY")
        print("="*60)
        print(f"Total Tests:     {summary['total_tests']}")
        print(f"Passed:          {summary['passed']} ✅")
        print(f"Failed:          {summary['failed']} ❌")
        print(f"Pass Rate:       {summary['pass_rate']:.1%}")
        print(f"Average Score:   {summary['avg_score']:.1%}")
        print(f"Min Score:       {summary['min_score']:.1%}")
        print(f"Max Score:       {summary['max_score']:.1%}")
        print(f"Avg Duration:    {summary['avg_duration_ms']:.0f}ms")
        print(f"Target (≥70%):   {'✅ MET' if summary['target_met'] else '❌ NOT MET'}")
        print("="*60)


# Sample test cases untuk validasi
SAMPLE_TEST_CASES = [
    {
        'query': 'Apa itu Kolabri dan bagaimana cara kerjanya?',
        'answer': 'Kolabri adalah platform AI untuk otomatisasi regulasi. Cara kerjanya adalah dengan menganalisis dokumen regulasi dan memberikan rekomendasi compliance.'
    },
    {
        'query': 'Bagaimana cara mengatur izin pengguna di sistem?',
        'answer': 'Untuk mengatur izin pengguna, Anda perlu masuk ke menu Settings > User Management > Permissions. Dari sana Anda dapat menambah, mengubah, atau menghapus izin.'
    },
    {
        'query': 'Apa saja fitur utama dari aplikasi ini?',
        'answer': 'Aplikasi ini memiliki fitur seperti compliance monitoring, document analysis, dan automated reporting yang membantu perusahaan mematuhi regulasi.'
    },
    {
        'query': 'Berapa biaya berlangganan untuk enterprise?',
        'answer': 'Untuk informasi harga enterprise, silakan hubungi tim sales kami di sales@kolabri.com atau isi form di halaman Contact Us.'
    },
    {
        'query': 'Apakah data saya aman di sistem ini?',
        'answer': 'Ya, kami menggunakan enkripsi AES-256 untuk data at rest dan TLS 1.3 untuk data in transit. Kami juga mematuhi standar keamanan ISO 27001.'
    }
]


async def main():
    """Main function untuk menjalankan relevance tests"""
    print("🎯 Answer Relevance Testing")
    print("Mengukur relevansi jawaban terhadap pertanyaan\n")

    tester = AnswerRelevanceTester()

    # Run tests with sample data
    summary = await tester.run_tests(SAMPLE_TEST_CASES)
    tester.print_summary(summary)

    # Save results
    output_file = 'relevance_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return summary


if __name__ == '__main__':
    asyncio.run(main())
