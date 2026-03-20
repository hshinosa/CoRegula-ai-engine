"""
RAG Faithfulness Test
=====================

Mengukur proporsi klaim dalam jawaban yang didukung oleh konteks.
Target: > 0.85

Metode:
1. Extract klaim dari jawaban RAG
2. Verifikasi setiap klaim terhadap konteks retrieval
3. Hitung rasio klaim yang terverifikasi
"""

import asyncio
import json
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import re


@dataclass
class FaithfulnessResult:
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    score: float
    details: List[Dict[str, Any]]


class FaithfulnessEvaluator:
    """Evaluator untuk faithfulness RAG."""
    
    def __init__(self):
        self.min_score = 0.85  # Target
    
    def extract_claims(self, answer: str) -> List[str]:
        """Extract klaim-klaim dari jawaban."""
        # Split berdasarkan kalimat
        sentences = re.split(r'[.!?]+', answer)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            # Filter kalimat yang mengandung fakta/informasi
            if len(sent) > 20 and not sent.lower().startswith(('bagaimana', 'mengapa', 'apakah')):
                claims.append(sent)
        
        return claims
    
    def verify_claim_against_context(self, claim: str, contexts: List[str]) -> Tuple[bool, str]:
        """Verifikasi satu klaim terhadap konteks."""
        claim_lower = claim.lower()
        
        for ctx in contexts:
            ctx_lower = ctx.lower()
            # Check keyword overlap
            claim_words = set(claim_lower.split())
            ctx_words = set(ctx_lower.split())
            
            overlap = len(claim_words & ctx_words)
            total_words = len(claim_words)
            
            if total_words > 0 and overlap / total_words >= 0.5:
                return True, ctx
        
        return False, ""
    
    async def evaluate(
        self,
        answer: str,
        contexts: List[str]
    ) -> FaithfulnessResult:
        """
        Evaluasi faithfulness jawaban RAG.
        
        Args:
            answer: Jawaban dari RAG
            contexts: Konteks yang di-retrieve
            
        Returns:
            FaithfulnessResult dengan score dan detail
        """
        claims = self.extract_claims(answer)
        
        if not claims:
            return FaithfulnessResult(
                total_claims=0,
                supported_claims=0,
                unsupported_claims=0,
                score=1.0,  # No claims = no hallucination
                details=[]
            )
        
        supported = 0
        unsupported = 0
        details = []
        
        for claim in claims:
            is_supported, evidence = self.verify_claim_against_context(claim, contexts)
            
            if is_supported:
                supported += 1
            else:
                unsupported += 1
            
            details.append({
                "claim": claim,
                "supported": is_supported,
                "evidence": evidence[:200] if evidence else None
            })
        
        score = supported / len(claims) if claims else 1.0
        
        return FaithfulnessResult(
            total_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported,
            score=score,
            details=details
        )
    
    def passed(self, result: FaithfulnessResult) -> bool:
        """Check jika score melewati target."""
        return result.score >= self.min_score


# Test cases
TEST_CASES = [
    {
        "name": "Basic ML Definition",
        "query": "Apa itu machine learning?",
        "contexts": [
            "Machine learning adalah cabang dari artificial intelligence yang memungkinkan sistem belajar dari data.",
            "ML menggunakan algoritma untuk mengidentifikasi pola dalam data dan membuat prediksi."
        ],
        "expected_claims": ["machine learning", "AI", "algoritma", "data"]
    },
    {
        "name": "Neural Network Explanation",
        "query": "Jelaskan neural network",
        "contexts": [
            "Neural network terinspirasi dari cara kerja otak manusia.",
            "Neural network terdiri dari neuron buatan yang terhubung dalam layer."
        ],
        "expected_claims": ["neural network", "otak", "neuron", "layer"]
    }
]


async def main():
    """Run faithfulness tests."""
    print("=" * 60)
    print("RAG FAITHFULNESS TEST")
    print("=" * 60)
    print(f"Target Score: > {FaithfulnessEvaluator().min_score}")
    print("=" * 60)
    
    evaluator = FaithfulnessEvaluator()
    
    for case in TEST_CASES:
        print(f"\nTest: {case['name']}")
        
        # Simulate RAG answer
        answer = f"Berdasarkan konteks, {case['query'].lower()} adalah " + \
                 " ".join([f"{claim}." for claim in case['expected_claims']])
        
        result = await evaluator.evaluate(answer, case['contexts'])
        
        print(f"  Total Claims: {result.total_claims}")
        print(f"  Supported: {result.supported_claims}")
        print(f"  Unsupported: {result.unsupported_claims}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Status: {'[PASS]' if evaluator.passed(result) else '[FAIL]'}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
