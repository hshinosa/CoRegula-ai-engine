"""
Tests for NLP Analytics Service - 100% Coverage
"""
import pytest
from app.services.nlp_analytics import (
    EngagementAnalyzer,
    EngagementType,
    EngagementAnalysis,
    get_engagement_analyzer,
)


class TestEngagementType:
    """Test EngagementType enum."""
    
    def test_engagement_type_values(self):
        """Test EngagementType enum values."""
        assert EngagementType.COGNITIVE.value == "cognitive"
        assert EngagementType.BEHAVIORAL.value == "behavioral"
        assert EngagementType.EMOTIONAL.value == "emotional"


class TestEngagementAnalysis:
    """Test EngagementAnalysis dataclass."""
    
    def test_engagement_analysis_creation(self):
        """Test creating EngagementAnalysis."""
        analysis = EngagementAnalysis(
            lexical_variety=0.65,
            engagement_type=EngagementType.COGNITIVE,
            is_higher_order=True,
            hot_indicators=["analyze", "evaluate"],
            word_count=50,
            unique_words=35,
            confidence=0.85
        )
        
        assert analysis.lexical_variety == 0.65
        assert analysis.engagement_type == EngagementType.COGNITIVE
        assert analysis.is_higher_order is True
        assert len(analysis.hot_indicators) == 2
        assert analysis.word_count == 50
        assert analysis.unique_words == 35
        assert analysis.confidence == 0.85


class TestEngagementAnalyzer:
    """Test EngagementAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return EngagementAnalyzer()
    
    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.hot_triggers is not None
        assert analyzer.emotional_words is not None
        assert analyzer.srl_keywords is not None
        assert len(analyzer.hot_triggers) > 0
    
    def test_extract_srl_object_found(self, analyzer):
        """Test extract_srl_object finds keywords."""
        text = "I'm learning about database and API design"
        result = analyzer.extract_srl_object(text)
        
        assert "Database" in result or "Api" in result
    
    def test_extract_srl_object_not_found(self, analyzer):
        """Test extract_srl_object returns default."""
        text = "This is a general discussion"
        result = analyzer.extract_srl_object(text)
        
        assert result == "General_Discussion"
    
    def test_extract_srl_object_multiple(self, analyzer):
        """Test extract_srl_object with multiple keywords."""
        text = "We use Python and FastAPI for the backend API"
        result = analyzer.extract_srl_object(text)
        
        # Should return up to 2 keywords
        assert len(result.split(", ")) <= 2
    
    def test_analyze_interaction_empty(self, analyzer):
        """Test analyze_interaction with empty text."""
        result = analyzer.analyze_interaction("")
        
        assert result.lexical_variety == 0.0
        assert result.engagement_type == EngagementType.BEHAVIORAL
        assert result.is_higher_order is False
        assert result.word_count == 0
        assert result.confidence == 0.0
    
    def test_analyze_interaction_whitespace(self, analyzer):
        """Test analyze_interaction with whitespace only."""
        result = analyzer.analyze_interaction("   ")
        
        assert result.lexical_variety == 0.0
        assert result.word_count == 0
    
    def test_analyze_interaction_simple(self, analyzer):
        """Test analyze_interaction with simple text."""
        result = analyzer.analyze_interaction("Hello world")
        
        assert result.word_count > 0
        assert result.lexical_variety >= 0.0
        assert result.lexical_variety <= 1.0
    
    def test_analyze_interaction_hot_detection(self, analyzer):
        """Test analyze_interaction detects HOT indicators."""
        text = "Saya menganalisis perbedaan antara database SQL dan NoSQL"
        result = analyzer.analyze_interaction(text)
        
        assert result.is_higher_order is True
        assert len(result.hot_indicators) > 0
        assert "analisis" in result.hot_indicators or "menganalisis" in result.hot_indicators
    
    def test_analyze_interaction_hot_english(self, analyzer):
        """Test analyze_interaction detects English HOT indicators."""
        text = "Let me analyze and evaluate this algorithm"
        result = analyzer.analyze_interaction(text)
        
        assert result.is_higher_order is True
        assert len(result.hot_indicators) > 0
    
    def test_analyze_interaction_cognitive(self, analyzer):
        """Test analyze_interaction classifies as cognitive."""
        text = "Saya mengevaluasi kelebihan dan kekurangan Redis untuk caching"
        result = analyzer.analyze_interaction(text)
        
        assert result.engagement_type == EngagementType.COGNITIVE
    
    def test_analyze_interaction_behavioral(self, analyzer):
        """Test analyze_interaction classifies as behavioral."""
        text = "Saya sudah menyelesaikan tugas ini"
        result = analyzer.analyze_interaction(text)
        
        # Should be behavioral (no HOT indicators)
        assert result.engagement_type == EngagementType.BEHAVIORAL
    
    def test_analyze_interaction_emotional(self, analyzer):
        """Test analyze_interaction classifies as emotional."""
        text = "Saya bingung dengan materi ini, sangat sulit"
        result = analyzer.analyze_interaction(text)
        
        assert result.engagement_type == EngagementType.EMOTIONAL
    
    def test_analyze_interaction_emotional_positive(self, analyzer):
        """Test analyze_interaction detects positive emotion."""
        text = "Saya senang dan tertarik dengan materi ini, sangat menarik!"
        result = analyzer.analyze_interaction(text)
        
        assert result.engagement_type == EngagementType.EMOTIONAL
    
    def test_analyze_interaction_lexical_variety_high(self, analyzer):
        """Test analyze_interaction with high lexical variety."""
        # Many unique words
        text = "Python JavaScript Java Go Rust CSharp Kotlin Swift TypeScript Ruby"
        result = analyzer.analyze_interaction(text)
        
        assert result.lexical_variety > 0.5
        assert result.unique_words > 5
    
    def test_analyze_interaction_lexical_variety_low(self, analyzer):
        """Test analyze_interaction with low lexical variety."""
        # Repeated words
        text = "saya saya saya saya saya saya"
        result = analyzer.analyze_interaction(text)
        
        assert result.lexical_variety < 0.5
    
    def test_analyze_interaction_confidence_short(self, analyzer):
        """Test confidence for short text."""
        result = analyzer.analyze_interaction("Hi")
        
        assert result.confidence <= 0.5
    
    def test_analyze_interaction_confidence_medium(self, analyzer):
        """Test confidence for medium text."""
        text = "Saya sedang belajar tentang database dan algoritma sorting" * 2
        result = analyzer.analyze_interaction(text)
        
        assert result.confidence >= 0.5
    
    def test_analyze_interaction_confidence_long(self, analyzer):
        """Test confidence for long text."""
        text = " ".join(["Saya belajar tentang Python dan FastAPI"] * 20)
        result = analyzer.analyze_interaction(text)
        
        assert result.confidence >= 0.7
    
    def test_analyze_interaction_hot_indicators_limited(self, analyzer):
        """Test that hot_indicators is limited to 5."""
        # Text with many HOT triggers
        text = " ".join(analyzer.HOT_TRIGGERS_ID[:20])
        result = analyzer.analyze_interaction(text)
        
        assert len(result.hot_indicators) <= 5
    
    def test_analyze_batch_empty(self, analyzer):
        """Test analyze_batch with empty list."""
        result = analyzer.analyze_batch([])
        
        assert result["count"] == 0
        assert result["avg_lexical_variety"] == 0.0
        assert result["hot_percentage"] == 0.0
        assert result["engagement_distribution"] == {}
        assert result["analyses"] == []
    
    def test_analyze_batch_single(self, analyzer):
        """Test analyze_batch with single text."""
        result = analyzer.analyze_batch(["Hello world"])
        
        assert result["count"] == 1
        assert len(result["analyses"]) == 1
    
    def test_analyze_batch_multiple(self, analyzer):
        """Test analyze_batch with multiple texts."""
        texts = [
            "Saya menganalisis database",
            "Saya menyelesaikan tugas",
            "Saya bingung dengan materi",
        ]
        result = analyzer.analyze_batch(texts)
        
        assert result["count"] == 3
        assert "avg_lexical_variety" in result
        assert "hot_percentage" in result
        assert "engagement_distribution" in result
        assert len(result["analyses"]) == 3
    
    def test_analyze_batch_mixed_hot(self, analyzer):
        """Test analyze_batch with mixed HOT indicators."""
        texts = [
            "Saya menganalisis dan mengevaluasi algoritma",  # HOT
            "Tugas selesai",  # Not HOT
        ]
        result = analyzer.analyze_batch(texts)
        
        assert result["hot_percentage"] > 0
        assert result["hot_percentage"] < 100
    
    def test_get_discussion_quality_score_empty(self, analyzer):
        """Test get_discussion_quality_score with empty texts."""
        result = analyzer.get_discussion_quality_score([])
        
        assert result["quality_score"] == 0.0
        assert "Belum ada data" in result["recommendation"]
    
    def test_get_discussion_quality_score_high(self, analyzer):
        """Test quality score with high quality texts."""
        texts = [
            "Saya menganalisis perbedaan antara SQL dan NoSQL",
            "Saya mengevaluasi kelebihan Redis untuk caching",
            "Saya merancang arsitektur sistem yang efisien",
        ]
        result = analyzer.get_discussion_quality_score(texts)
        
        assert result["quality_score"] > 0
        assert "lexical_score" in result
        assert "hot_score" in result
        assert "cognitive_ratio" in result
    
    def test_get_discussion_quality_score_components(self, analyzer):
        """Test quality score returns all components."""
        texts = ["Test message"]
        result = analyzer.get_discussion_quality_score(texts)
        
        assert "quality_score" in result
        assert "lexical_score" in result
        assert "hot_score" in result
        assert "cognitive_ratio" in result
        assert "recommendation" in result
    
    def test_get_quality_recommendation_high(self, analyzer):
        """Test recommendation for high score."""
        result = analyzer._get_quality_recommendation(75)
        assert "tinggi" in result.lower() or "pertahankan" in result.lower()
    
    def test_get_quality_recommendation_medium_high(self, analyzer):
        """Test recommendation for medium-high score."""
        result = analyzer._get_quality_recommendation(60)
        assert "baik" in result.lower() or "tingkatkan" in result.lower()
    
    def test_get_quality_recommendation_medium_low(self, analyzer):
        """Test recommendation for medium-low score."""
        result = analyzer._get_quality_recommendation(40)
        assert "ditingkatkan" in result.lower() or "analisis" in result.lower()
    
    def test_get_quality_recommendation_low(self, analyzer):
        """Test recommendation for low score."""
        result = analyzer._get_quality_recommendation(20)
        assert "dangkal" in result.lower() or "intervensi" in result.lower()
    
    def test_tokenize(self, analyzer):
        """Test _tokenize method."""
        result = analyzer._tokenize("Hello, World! Test.")
        
        assert "Hello" in result
        assert "World" in result
        assert "Test" in result
        assert len(result) == 3
    
    def test_tokenize_empty(self, analyzer):
        """Test _tokenize with empty string."""
        result = analyzer._tokenize("")
        assert result == []
    
    def test_calculate_lexical_variety_empty(self, analyzer):
        """Test lexical variety with empty words."""
        result = analyzer._calculate_lexical_variety([])
        assert result == 0.0
    
    def test_calculate_lexical_variety_short(self, analyzer):
        """Test lexical variety with short text."""
        words = ["hello", "world", "hello"]
        result = analyzer._calculate_lexical_variety(words)
        
        # 2 unique / 3 total = 0.667
        assert result > 0.6
        assert result <= 1.0
    
    def test_calculate_lexical_variety_long(self, analyzer):
        """Test lexical variety with long text (>50 words)."""
        words = ["word"] * 100  # All same
        result = analyzer._calculate_lexical_variety(words)
        
        # Uses sqrt normalization for long texts
        assert result >= 0.0
        assert result <= 1.0
    
    def test_detect_hot_indicators_found(self, analyzer):
        """Test HOT detection finds indicators."""
        text = "saya menganalisis dan mengevaluasi algoritma"
        result = analyzer._detect_hot_indicators(text)
        
        assert len(result) > 0
        assert "menganalisis" in result or "analisis" in result
    
    def test_detect_hot_indicators_not_found(self, analyzer):
        """Test HOT detection with no indicators."""
        text = "hello world test"
        result = analyzer._detect_hot_indicators(text)
        
        assert len(result) == 0
    
    def test_detect_hot_indicators_limited(self, analyzer):
        """Test HOT detection limits to 5."""
        # Text with many triggers
        text = " ".join(analyzer.HOT_TRIGGERS_ID[:20])
        result = analyzer._detect_hot_indicators(text)
        
        assert len(result) <= 5
    
    def test_classify_engagement_emotional(self, analyzer):
        """Test engagement classification as emotional."""
        text = "saya bingung dan frustasi"
        result = analyzer._classify_engagement(text, False, [])
        
        assert result == EngagementType.EMOTIONAL
    
    def test_classify_engagement_cognitive_hot(self, analyzer):
        """Test engagement classification as cognitive with HOT."""
        text = "algoritma sorting"
        result = analyzer._classify_engagement(text, True, ["analyze"])
        
        assert result == EngagementType.COGNITIVE
    
    def test_classify_engagement_cognitive_indicators(self, analyzer):
        """Test engagement classification as cognitive with indicators."""
        text = "test"
        result = analyzer._classify_engagement(text, False, ["analyze", "evaluate"])
        
        assert result == EngagementType.COGNITIVE
    
    def test_classify_engagement_behavioral(self, analyzer):
        """Test engagement classification as behavioral."""
        text = "saya mengerjakan tugas"
        result = analyzer._classify_engagement(text, False, [])
        
        assert result == EngagementType.BEHAVIORAL
    
    def test_calculate_confidence_very_short(self, analyzer):
        """Test confidence for very short text (<=5)."""
        result = analyzer._calculate_confidence(3, 2, False)
        assert result == 0.3
    
    def test_calculate_confidence_short(self, analyzer):
        """Test confidence for short text (<=15)."""
        result = analyzer._calculate_confidence(10, 8, False)
        assert result == 0.5
    
    def test_calculate_confidence_medium(self, analyzer):
        """Test confidence for medium text (<=50)."""
        result = analyzer._calculate_confidence(30, 20, False)
        assert result == 0.7
    
    def test_calculate_confidence_long(self, analyzer):
        """Test confidence for long text (>50)."""
        result = analyzer._calculate_confidence(100, 50, True)
        assert result == 0.9
    
    def test_to_dict(self, analyzer):
        """Test _to_dict conversion."""
        analysis = EngagementAnalysis(
            lexical_variety=0.65,
            engagement_type=EngagementType.COGNITIVE,
            is_higher_order=True,
            hot_indicators=["analyze"],
            word_count=50,
            unique_words=35,
            confidence=0.85
        )
        
        result = analyzer._to_dict(analysis)
        
        assert result["lexical_variety"] == 0.65
        assert result["engagement_type"] == "cognitive"
        assert result["is_higher_order"] is True
        assert result["hot_indicators"] == ["analyze"]
        assert result["word_count"] == 50
        assert result["unique_words"] == 35
        assert result["confidence"] == 0.85


class TestGetEngagementAnalyzer:
    """Test get_engagement_analyzer singleton."""
    
    def test_get_engagement_analyzer_singleton(self):
        """Test get_engagement_analyzer returns singleton."""
        from app.services import nlp_analytics
        nlp_analytics._analyzer = None
        
        analyzer1 = get_engagement_analyzer()
        analyzer2 = get_engagement_analyzer()
        
        assert analyzer1 is analyzer2
        assert isinstance(analyzer1, EngagementAnalyzer)
    
    def test_get_engagement_analyzer_initialization(self):
        """Test get_engagement_analyzer initializes correctly."""
        from app.services import nlp_analytics
        nlp_analytics._analyzer = None
        
        analyzer = get_engagement_analyzer()
        
        assert analyzer is not None
        assert isinstance(analyzer, EngagementAnalyzer)
        assert len(analyzer.hot_triggers) > 0
