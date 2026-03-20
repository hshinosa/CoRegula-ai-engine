import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.efficiency_guard import EfficiencyGuard, CacheEntry, RateLimiter, QueryDeduplicator, get_efficiency_guard

@pytest.fixture
def guard():
    return EfficiencyGuard(max_cache_size=2)

@pytest.mark.unit
def test_cache_entry():
    entry = CacheEntry(response={'r': 1}, ttl_seconds=10)
    assert entry.is_expired() is False
    with patch('app.services.efficiency_guard.datetime') as mock_datetime:
        mock_datetime.now.return_value = entry.created_at + timedelta(seconds=11)
        assert entry.is_expired() is True
    assert entry.to_dict()['hit_count'] == 0

@pytest.mark.unit
def test_rate_limiter():
    limiter = RateLimiter(max_requests=1, time_window_seconds=60)
    assert limiter.is_allowed('u1') is True
    assert limiter.is_allowed('u1') is False
    with patch('app.services.efficiency_guard.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime.now() + timedelta(seconds=61)
        assert limiter.is_allowed('u1') is True

@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_deduplicator():
    dedup = QueryDeduplicator()
    event = asyncio.Event()
    async def mock_func():
        await event.wait()
        return 'res'
    task1 = asyncio.create_task(dedup.execute_or_wait('k1', mock_func))
    task2 = asyncio.create_task(dedup.execute_or_wait('k1', mock_func))
    await asyncio.sleep(0.1)
    event.set()
    res = await asyncio.gather(task1, task2)
    assert res == ['res', 'res']

@pytest.mark.unit
@pytest.mark.asyncio
async def test_efficiency_guard_full_coverage(guard):
    await guard.cache_response('k1', 'v1')
    assert await guard.get_cached_response('k1') == 'v1'
    guard.query_patterns['q1'] = 10
    stats = guard.get_statistics()
    assert stats['cache']['cache_size'] == 1
    hf = guard.get_high_frequency_queries(limit=1)
    assert len(hf) > 0
    await guard.clear_cache()
    await guard.clear_expired_cache()
    assert len(guard.cache) == 0

@pytest.mark.unit
def test_singleton():
    v1 = get_efficiency_guard()
    v2 = get_efficiency_guard()
    assert v1 is v2