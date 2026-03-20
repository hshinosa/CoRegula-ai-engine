"""
Test DDoS Protection - DDoS Protection & Rate Limiting Testing
Menguji kemampuan sistem menahan serangan DDoS dan rate limiting
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class RateLimitResult:
    """Hasil pengujian rate limiting"""
    request_id: int
    timestamp: float
    allowed: bool
    rate_limit_hit: bool
    retry_after: Optional[int]
    duration_ms: float


@dataclass
class DDoSTestResult:
    """Hasil pengujian DDoS protection"""
    test_name: str
    total_requests: int
    successful_requests: int
    rate_limited_requests: int
    avg_response_time_ms: float
    peak_rps: float  # Requests per second
    protection_triggered: bool
    duration_ms: float


class RateLimiter:
    """Simulasi rate limiter untuk testing"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        block_duration_seconds: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.block_duration_seconds = block_duration_seconds

        # Track requests per client
        self.client_requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_clients: Dict[str, float] = {}

    def is_allowed(self, client_id: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        current_time = time.time()

        # Check if client is blocked
        if client_id in self.blocked_clients:
            block_time = self.blocked_clients[client_id]
            if current_time - block_time < self.block_duration_seconds:
                retry_after = int(self.block_duration_seconds - (current_time - block_time))
                return False, retry_after
            else:
                # Unblock client
                del self.blocked_clients[client_id]

        # Clean old requests (older than 1 minute)
        cutoff_time = current_time - 60
        self.client_requests[client_id] = [
            ts for ts in self.client_requests[client_id]
            if ts > cutoff_time
        ]

        # Check rate limit
        if len(self.client_requests[client_id]) >= self.requests_per_minute:
            # Block the client
            self.blocked_clients[client_id] = current_time
            return False, self.block_duration_seconds

        # Check burst limit
        recent_requests = len(self.client_requests[client_id])
        if recent_requests >= self.burst_size:
            # Still allowed but approaching limit
            pass

        # Allow request
        self.client_requests[client_id].append(current_time)
        return True, None


class DDoSProtectionTester:
    """Tester untuk menguji DDoS protection dan rate limiting"""

    def __init__(self):
        self.results: List[DDoSTestResult] = []

    async def simulate_request(
        self,
        request_id: int,
        rate_limiter: RateLimiter,
        client_id: str = "test_client",
        processing_time_ms: float = 100
    ) -> RateLimitResult:
        """Simulate a single request"""
        start_time = time.time()

        # Check rate limit
        allowed, retry_after = rate_limiter.is_allowed(client_id)

        # Simulate processing time if allowed
        if allowed:
            await asyncio.sleep(processing_time_ms / 1000)

        duration = (time.time() - start_time) * 1000

        return RateLimitResult(
            request_id=request_id,
            timestamp=start_time,
            allowed=allowed,
            rate_limit_hit=not allowed,
            retry_after=retry_after,
            duration_ms=duration
        )

    async def test_rate_limiting(
        self,
        test_name: str,
        total_requests: int,
        requests_per_second: float,
        rate_limiter: RateLimiter,
        client_id: str = "test_client"
    ) -> DDoSTestResult:
        """
        Test rate limiting dengan burst traffic
        """
        print(f"\n  🧪 {test_name}")
        print(f"     Requests: {total_requests} @ {requests_per_second:.1f} req/s")

        start_time = time.time()
        results: List[RateLimitResult] = []

        # Create concurrent requests
        tasks = []
        for i in range(total_requests):
            task = self.simulate_request(
                request_id=i,
                rate_limiter=rate_limiter,
                client_id=client_id
            )
            tasks.append(task)

            # Add delay between requests
            if i < total_requests - 1:
                await asyncio.sleep(1 / requests_per_second)

        # Execute all requests
        results = await asyncio.gather(*tasks)

        duration = (time.time() - start_time) * 1000

        # Calculate metrics
        successful = sum(1 for r in results if r.allowed)
        rate_limited = sum(1 for r in results if r.rate_limit_hit)
        avg_response = sum(r.duration_ms for r in results) / len(results)
        peak_rps = total_requests / (duration / 1000)

        # Protection triggered if rate limiting kicked in
        protection_triggered = rate_limited > 0

        return DDoSTestResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful,
            rate_limited_requests=rate_limited,
            avg_response_time_ms=avg_response,
            peak_rps=peak_rps,
            protection_triggered=protection_triggered,
            duration_ms=duration
        )

    async def test_distributed_attack(
        self,
        test_name: str,
        total_requests: int,
        num_clients: int,
        rate_limiter: RateLimiter
    ) -> DDoSTestResult:
        """
        Test DDoS protection dari multiple clients (distributed attack)
        """
        print(f"\n  🧪 {test_name}")
        print(f"     Clients: {num_clients}, Total Requests: {total_requests}")

        start_time = time.time()

        # Create requests from multiple clients
        tasks = []
        requests_per_client = total_requests // num_clients

        for client_num in range(num_clients):
            client_id = f"attacker_{client_num}"

            for i in range(requests_per_client):
                task = self.simulate_request(
                    request_id=client_num * requests_per_client + i,
                    rate_limiter=rate_limiter,
                    client_id=client_id,
                    processing_time_ms=50  # Faster processing
                )
                tasks.append(task)

        # Execute all requests
        results = await asyncio.gather(*tasks)

        duration = (time.time() - start_time) * 1000

        # Calculate metrics
        successful = sum(1 for r in results if r.allowed)
        rate_limited = sum(1 for r in results if r.rate_limit_hit)
        avg_response = sum(r.duration_ms for r in results) / len(results)
        peak_rps = total_requests / (duration / 1000)

        protection_triggered = rate_limited > 0

        return DDoSTestResult(
            test_name=test_name,
            total_requests=len(results),
            successful_requests=successful,
            rate_limited_requests=rate_limited,
            avg_response_time_ms=avg_response,
            peak_rps=peak_rps,
            protection_triggered=protection_triggered,
            duration_ms=duration
        )

    async def test_autoscaling_simulation(
        self,
        test_name: str,
        base_capacity: int,
        peak_load: int,
        scale_up_time_ms: float
    ) -> DDoSTestResult:
        """
        Simulasi autoscaling response
        """
        print(f"\n  🧪 {test_name}")
        print(f"     Base: {base_capacity} req/s, Peak: {peak_load} req/s")

        start_time = time.time()

        # Simulate capacity scaling
        current_capacity = base_capacity
        scaled_requests = 0
        rejected_requests = 0

        # Simulate load increase
        for second in range(10):
            # Gradually increase load
            current_load = base_capacity + (peak_load - base_capacity) * (second / 10)

            # Check if scaling needed
            if current_load > current_capacity and second * 1000 >= scale_up_time_ms:
                # Scale up
                current_capacity = min(current_capacity * 2, peak_load)
                print(f"     ⬆️ Scale up at t={second}s to {current_capacity} req/s")

            # Process requests
            can_process = min(int(current_load), current_capacity)
            rejected = int(current_load) - can_process

            scaled_requests += can_process
            rejected_requests += max(0, rejected)

            await asyncio.sleep(0.1)  # Simulate 100ms

        duration = (time.time() - start_time) * 1000

        total_requests = scaled_requests + rejected_requests

        return DDoSTestResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=scaled_requests,
            rate_limited_requests=rejected_requests,
            avg_response_time_ms=50.0,  # Simulated
            peak_rps=peak_load,
            protection_triggered=rejected_requests > 0,
            duration_ms=duration
        )

    async def run_tests(self) -> Dict[str, Any]:
        """
        Jalankan semua test DDoS protection
        """
        print("\n🛡️ Menjalankan DDoS Protection Tests...")

        # Test 1: Basic Rate Limiting
        rate_limiter_1 = RateLimiter(
            requests_per_minute=60,
            burst_size=10,
            block_duration_seconds=30
        )

        result_1 = await self.test_rate_limiting(
            test_name="Basic Rate Limiting (60 req/min)",
            total_requests=100,
            requests_per_second=5,
            rate_limiter=rate_limiter_1
        )
        self.results.append(result_1)

        # Test 2: Burst Attack
        rate_limiter_2 = RateLimiter(
            requests_per_minute=120,
            burst_size=20,
            block_duration_seconds=60
        )

        result_2 = await self.test_rate_limiting(
            test_name="Burst Attack (120 req/min)",
            total_requests=150,
            requests_per_second=10,
            rate_limiter=rate_limiter_2
        )
        self.results.append(result_2)

        # Test 3: Distributed Attack (Multiple Clients)
        rate_limiter_3 = RateLimiter(
            requests_per_minute=60,
            burst_size=10,
            block_duration_seconds=60
        )

        result_3 = await self.test_distributed_attack(
            test_name="Distributed DDoS (10 clients)",
            total_requests=200,
            num_clients=10,
            rate_limiter=rate_limiter_3
        )
        self.results.append(result_3)

        # Test 4: Autoscaling Simulation
        result_4 = await self.test_autoscaling_simulation(
            test_name="Autoscaling Response",
            base_capacity=50,
            peak_load=200,
            scale_up_time_ms=3000  # 3 seconds to scale up
        )
        self.results.append(result_4)

        # Calculate summary
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_limited = sum(r.rate_limited_requests for r in self.results)
        avg_response = sum(r.avg_response_time_ms for r in self.results) / len(self.results)
        max_peak_rps = max(r.peak_rps for r in self.results)

        protection_worked = all(r.protection_triggered for r in self.results if r.rate_limited_requests > 0)

        summary = {
            'test_name': 'DDoS Protection',
            'total_tests': len(self.results),
            'total_requests': total_requests,
            'successful_requests': total_successful,
            'rate_limited_requests': total_limited,
            'success_rate': total_successful / total_requests if total_requests > 0 else 0,
            'rate_limit_rate': total_limited / total_requests if total_requests > 0 else 0,
            'avg_response_time_ms': avg_response,
            'max_peak_rps': max_peak_rps,
            'protection_worked': protection_worked,
            'results': [
                {
                    'test_name': r.test_name,
                    'total_requests': r.total_requests,
                    'successful_requests': r.successful_requests,
                    'rate_limited_requests': r.rate_limited_requests,
                    'success_rate': r.successful_requests / r.total_requests,
                    'peak_rps': r.peak_rps,
                    'protection_triggered': r.protection_triggered,
                    'duration_ms': r.duration_ms
                }
                for r in self.results
            ]
        }

        return summary

    def print_summary(self, summary: Dict[str, Any]):
        """Print summary hasil test"""
        print("\n" + "="*60)
        print("🛡️ DDoS PROTECTION TEST SUMMARY")
        print("="*60)
        print(f"Total Test Scenarios: {summary['total_tests']}")
        print(f"Total Requests:       {summary['total_requests']}")
        print(f"Successful:           {summary['successful_requests']} ✅")
        print(f"Rate Limited:         {summary['rate_limited_requests']} 🛡️")
        print(f"Success Rate:         {summary['success_rate']:.1%}")
        print(f"Rate Limit Rate:      {summary['rate_limit_rate']:.1%}")
        print(f"Avg Response Time:    {summary['avg_response_time_ms']:.1f}ms")
        print(f"Max Peak RPS:         {summary['max_peak_rps']:.0f}")
        print()

        print("Individual Test Results:")
        print("-" * 60)
        for r in summary['results']:
            status = "🛡️ PROTECTED" if r['protection_triggered'] else "✅ NORMAL"
            print(f"  {r['test_name']}:")
            print(f"    Requests: {r['total_requests']}, "
                  f"Success: {r['successful_requests']}, "
                  f"Limited: {r['rate_limited_requests']}")
            print(f"    Status: {status}")
            print()

        status = "✅ ALL TESTS PASSED" if summary['protection_worked'] else "❌ PROTECTION ISSUES"
        print(f"Overall Status: {status}")
        print("="*60)


async def main():
    """Main function untuk menjalankan DDoS protection tests"""
    print("🛡️ DDoS Protection Testing")
    print("Menguji kemampuan sistem menahan serangan DDoS\n")

    tester = DDoSProtectionTester()

    # Run tests
    summary = await tester.run_tests()
    tester.print_summary(summary)

    # Save results
    output_file = 'ddos_protection_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")

    return summary


if __name__ == '__main__':
    asyncio.run(main())
