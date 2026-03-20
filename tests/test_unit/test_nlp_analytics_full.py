import pytest
from app.services.nlp_analytics import EngagementAnalyzer, EngagementType, EngagementAnalysis

@pytest.fixture
def analyzer():
    return EngagementAnalyzer()

@pytest.mark.unit
def test_analyze_interaction_full(analyzer):
    # Test all engagement types
    cognitive = 'Saya ingin menganalisis dampak redis caching.'
    res1 = analyzer.analyze_interaction(cognitive)
    assert res1.engagement_type == EngagementType.COGNITIVE
    
    behavioral = 'Oke baik.'
    res2 = analyzer.analyze_interaction(behavioral)
    assert res2.engagement_type == EngagementType.BEHAVIORAL
    
    emotional = 'Saya bingung sekali, sangat sulit.'
    res3 = analyzer.analyze_interaction(emotional)
    assert res3.engagement_type == EngagementType.EMOTIONAL

@pytest.mark.unit
def test_quality_score_edge_cases(analyzer):
    # Empty texts
    res1 = analyzer.get_discussion_quality_score([])
    assert res1['quality_score'] == 0.0
    
    # Very short text
    res2 = analyzer.get_discussion_quality_score(['Hi'])
    assert res2['quality_score'] >= 0.0

@pytest.mark.unit
def test_tokenize_edge_cases(analyzer):
    # Line 186: Empty after cleaning
    assert analyzer._tokenize('!!!') == []