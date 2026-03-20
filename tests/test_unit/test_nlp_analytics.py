import pytest
from app.services.nlp_analytics import EngagementAnalyzer, EngagementType, EngagementAnalysis

@pytest.fixture
def analyzer():
    return EngagementAnalyzer()

@pytest.mark.unit
def test_analyze_interaction_basic(analyzer):
    text = 'Saya ingin menganalisis dampak redis caching pada performa API.'
    res = analyzer.analyze_interaction(text)
    assert res.engagement_type == EngagementType.COGNITIVE
    assert res.is_higher_order is True
    assert 'menganalisis' in res.hot_indicators
    assert res.word_count > 0

@pytest.mark.unit
def test_analyze_interaction_emotional(analyzer):
    text = 'Saya bingung sekali dengan materi ini, sangat sulit.'
    res = analyzer.analyze_interaction(text)
    assert res.engagement_type == EngagementType.EMOTIONAL
    assert 'bingung' in text or 'sulit' in text

@pytest.mark.unit
def test_analyze_interaction_behavioral(analyzer):
    text = 'Oke baik.'
    res = analyzer.analyze_interaction(text)
    assert res.engagement_type == EngagementType.BEHAVIORAL
    assert res.is_higher_order is False

@pytest.mark.unit
def test_extract_srl_object(analyzer):
    text = 'Bagaimana cara setup mongodb dan redis?'
    res = analyzer.extract_srl_object(text)
    assert 'Mongodb' in res or 'Redis' in res

@pytest.mark.unit
def test_get_discussion_quality_score(analyzer):
    texts = [
        'Saya ingin menganalisis ini.',
        'Mengapa hal ini terjadi?',
        'Menurut saya ini menarik.'
    ]
    res = analyzer.get_discussion_quality_score(texts)
    assert res['quality_score'] > 0
    assert 'recommendation' in res

@pytest.mark.unit
def test_analyze_batch(analyzer):
    texts = ['Pesan 1', 'Pesan 2']
    res = analyzer.analyze_batch(texts)
    assert res['count'] == 2
    assert len(res['analyses']) == 2

@pytest.mark.unit
def test_empty_inputs(analyzer):
    assert analyzer.analyze_interaction('') is not None
    assert analyzer.analyze_batch([])['count'] == 0
    assert analyzer.get_discussion_quality_score([])['quality_score'] == 0.0