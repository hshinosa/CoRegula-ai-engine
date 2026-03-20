"""
Tests for Text Processor - 100% Coverage
"""
import pytest
from app.utils.text_processor import normalize_text, clean_for_ttr


class TestNormalizeText:
    """Test normalize_text function."""
    
    def test_normalize_text_empty(self):
        """Test normalize_text with empty string."""
        assert normalize_text("") == ""
    
    def test_normalize_text_none(self):
        """Test normalize_text with None."""
        assert normalize_text(None) == ""
    
    def test_normalize_text_basic(self):
        """Test normalize_text with basic text."""
        text = "Hello World"
        result = normalize_text(text)
        assert result == "Hello World"
    
    def test_normalize_text_extra_whitespace(self):
        """Test normalize_text with extra whitespace."""
        text = "Hello    World   Test"
        result = normalize_text(text)
        assert result == "Hello World Test"
    
    def test_normalize_text_unicode(self):
        """Test normalize_text with unicode characters."""
        text = "Héllo Wörld"
        result = normalize_text(text)
        assert result == "Héllo Wörld"
    
    def test_normalize_text_special_chars_removed(self):
        """Test normalize_text removes special characters."""
        text = "Hello@World#Test$"
        result = normalize_text(text)
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
    
    def test_normalize_text_keeps_punctuation(self):
        """Test normalize_text keeps basic punctuation."""
        text = "Hello, World! How are you?"
        result = normalize_text(text)
        assert "," in result
        assert "!" in result
        assert "?" in result
    
    def test_normalize_text_keeps_parentheses(self):
        """Test normalize_text keeps parentheses."""
        text = "Hello (World) Test"
        result = normalize_text(text)
        assert "(" in result
        assert ")" in result
    
    def test_normalize_text_keeps_hyphen(self):
        """Test normalize_text keeps hyphen."""
        text = "Hello-World Test"
        result = normalize_text(text)
        assert "-" in result
    
    def test_normalize_text_strip(self):
        """Test normalize_text strips leading/trailing whitespace."""
        text = "  Hello World  "
        result = normalize_text(text)
        assert result == "Hello World"
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestCleanForTtr:
    """Test clean_for_ttr function."""
    
    def test_clean_for_ttr_empty(self):
        """Test clean_for_ttr with empty string."""
        assert clean_for_ttr("") == ""
    
    def test_clean_for_ttr_lowercase(self):
        """Test clean_for_ttr converts to lowercase."""
        text = "Hello WORLD"
        result = clean_for_ttr(text)
        assert result == "hello world"
    
    def test_clean_for_ttr_remove_punctuation(self):
        """Test clean_for_ttr removes all punctuation."""
        text = "Hello, World! Test?"
        result = clean_for_ttr(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result
    
    def test_clean_for_ttr_keeps_spaces(self):
        """Test clean_for_ttr keeps spaces."""
        text = "Hello World Test"
        result = clean_for_ttr(text)
        assert " " in result
    
    def test_clean_for_ttr_complex(self):
        """Test clean_for_ttr with complex text."""
        text = "Hello, World! How are you? I'm fine."
        result = clean_for_ttr(text)
        assert result == "hello world how are you im fine"
    
    def test_clean_for_ttr_numbers(self):
        """Test clean_for_ttr keeps numbers."""
        text = "Test 123 ABC"
        result = clean_for_ttr(text)
        assert result == "test 123 abc"
    
    def test_clean_for_ttr_special_chars(self):
        """Test clean_for_ttr removes special characters."""
        text = "Hello@World#Test$123"
        result = clean_for_ttr(text)
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert result == "helloworldtest123"
