"""
Gemini LLM Service - MVP Phase 1-1.5
CoRegula AI Engine

Provides LLM capabilities using Google Gemini for:
- Chat completions
- RAG-enhanced responses
- Intervention text generation
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "user" or "model"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    tokens_used: int
    model: str
    success: bool
    error: Optional[str] = None


class GeminiLLMService:
    """
    Service for interacting with Google Gemini LLM.
    
    Features:
    - Chat completion with context
    - RAG-enhanced responses
    - Configurable generation parameters
    - Retry logic for reliability
    """
    
    # System prompts for different contexts
    SYSTEM_PROMPTS = {
        "default": """Anda adalah asisten AI CoRegula yang membantu mahasiswa dalam perkuliahan.
Anda harus memberikan jawaban yang akurat, informatif, dan mendukung pembelajaran.
Gunakan bahasa Indonesia yang baik dan mudah dipahami.""",
        
        "rag": """Anda adalah asisten AI CoRegula yang membantu mahasiswa memahami materi perkuliahan.
    Sampaikan jawaban seolah berasal dari pengetahuan Anda sendiri yang didukung informasi terkurasi.
    Jelaskan ruang lingkup pengetahuan tersebut, sebutkan keterbatasan saat informasi tidak tersedia, dan jangan merujuk ke dokumen atau sumber spesifik.
    Berikan penjelasan yang jelas dan terstruktur.""",
        
        "intervention": """Anda adalah fasilitator diskusi yang membantu menjaga kualitas percakapan akademik.
Tugas Anda adalah memberikan intervensi yang konstruktif untuk:
- Mengarahkan diskusi kembali ke topik
- Memberikan prompt yang memicu pemikiran kritis
- Menyarankan sumber belajar yang relevan
- Merangkum poin-poin penting dari diskusi
Gunakan nada yang ramah dan mendukung.""",
        
        "summary": """Anda adalah asisten yang ahli dalam merangkum diskusi akademik.
Buat ringkasan yang mencakup:
- Poin-poin utama yang dibahas
- Kesimpulan yang dicapai
- Pertanyaan yang belum terjawab
- Rekomendasi untuk tindak lanjut"""
    }
    
    def __init__(self):
        """Initialize Gemini LLM service."""
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        self.model_name = settings.GEMINI_MODEL
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self._get_default_config()
        )
        
        logger.info(
            "gemini_llm_initialized",
            model=self.model_name
        )
    
    def _get_default_config(self) -> GenerationConfig:
        """Get default generation configuration."""
        return GenerationConfig(
            temperature=settings.GEMINI_TEMPERATURE,
            top_p=settings.GEMINI_TOP_P,
            top_k=40,
            max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt override
            context: Optional context (for RAG)
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            LLMResponse with generated content
        """
        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, system_prompt, context)
            
            # Configure generation if overrides provided
            config = GenerationConfig(
                temperature=temperature or settings.GEMINI_TEMPERATURE,
                top_p=settings.GEMINI_TOP_P,
                top_k=40,
                max_output_tokens=max_tokens or settings.GEMINI_MAX_OUTPUT_TOKENS,
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=config
            )
            
            # Extract token count (approximate)
            tokens_used = self._estimate_tokens(full_prompt, response.text)
            
            logger.info(
                "llm_generation_complete",
                prompt_length=len(prompt),
                response_length=len(response.text),
                tokens_used=tokens_used
            )
            
            return LLMResponse(
                content=response.text,
                tokens_used=tokens_used,
                model=self.model_name,
                success=True
            )
            
        except Exception as e:
            logger.error(
                "llm_generation_failed",
                error=str(e),
                prompt_preview=prompt[:100]
            )
            return LLMResponse(
                content="",
                tokens_used=0,
                model=self.model_name,
                success=False,
                error=str(e)
            )
    
    async def generate_rag_response(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        chat_history: Optional[List[ChatMessage]] = None
    ) -> LLMResponse:
        """
        Generate a RAG-enhanced response.
        
        Args:
            query: User query
            contexts: Retrieved context documents
            chat_history: Optional chat history for context
            
        Returns:
            LLMResponse with contextual answer
        """
        # Format contexts
        context_text = self._format_contexts(contexts)
        
        # Build RAG prompt
        rag_prompt = f"""Konteks pengetahuan yang tersedia:
{context_text}

Pertanyaan: {query}

    Gunakan pengetahuan di atas untuk menjawab. 
    Jika pengetahuan tersebut tidak cukup, jelaskan keterbatasan Anda dan berikan informasi umum yang relevan."""
        
        # Include chat history if available
        if chat_history:
            history_text = self._format_chat_history(chat_history)
            rag_prompt = f"""Riwayat percakapan:
{history_text}

{rag_prompt}"""
        
        return await self.generate(
            prompt=rag_prompt,
            system_prompt=self.SYSTEM_PROMPTS["rag"]
        )
    
    async def generate_intervention(
        self,
        chat_messages: List[Dict[str, Any]],
        intervention_type: str = "redirect",
        topic: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate an intervention message for chat.
        
        Args:
            chat_messages: Recent chat messages
            intervention_type: Type of intervention (redirect, prompt, summarize)
            topic: Optional topic to redirect to
            
        Returns:
            LLMResponse with intervention message
        """
        # Format chat messages
        chat_text = "\n".join([
            f"{msg.get('sender', 'User')}: {msg.get('content', '')}"
            for msg in chat_messages[-10:]  # Last 10 messages
        ])
        
        # Build intervention prompt based on type
        if intervention_type == "redirect":
            prompt = f"""Berikut adalah percakapan terakhir dalam grup diskusi:

{chat_text}

Topik seharusnya: {topic or 'materi perkuliahan'}

Buatkan intervensi yang ramah untuk mengarahkan diskusi kembali ke topik. 
Maksimal 2-3 kalimat."""
            
        elif intervention_type == "prompt":
            prompt = f"""Berikut adalah percakapan terakhir dalam grup diskusi:

{chat_text}

Topik diskusi: {topic or 'materi perkuliahan'}

Buatkan pertanyaan pemicu yang mendorong diskusi lebih dalam dan pemikiran kritis.
Maksimal 1-2 pertanyaan."""
            
        elif intervention_type == "summarize":
            prompt = f"""Berikut adalah percakapan dalam grup diskusi:

{chat_text}

Buatkan ringkasan singkat dari diskusi ini mencakup:
- Poin utama yang dibahas
- Kesimpulan (jika ada)
- Saran untuk tindak lanjut"""
            
        else:
            prompt = f"""Berikut adalah percakapan terakhir:

{chat_text}

Berikan komentar yang membantu dan relevan untuk mendukung diskusi."""
        
        return await self.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPTS["intervention"],
            temperature=0.7  # Slightly more creative for interventions
        )
    
    async def generate_summary(
        self,
        messages: List[Dict[str, Any]],
        include_action_items: bool = True
    ) -> LLMResponse:
        """
        Generate a summary of chat messages.
        
        Args:
            messages: Chat messages to summarize
            include_action_items: Whether to include action items
            
        Returns:
            LLMResponse with summary
        """
        # Format messages
        messages_text = "\n".join([
            f"{msg.get('sender', 'User')}: {msg.get('content', '')}"
            for msg in messages
        ])
        
        prompt = f"""Berikut adalah percakapan yang perlu dirangkum:

{messages_text}

Buatkan ringkasan yang mencakup:
1. Poin-poin utama yang dibahas
2. Kesimpulan yang dicapai
3. Pertanyaan yang belum terjawab"""
        
        if include_action_items:
            prompt += "\n4. Action items atau tindak lanjut yang disarankan"
        
        return await self.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPTS["summary"]
        )
    
    def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """Build the complete prompt with system instructions and context."""
        parts = []
        
        # Add system prompt
        if system_prompt:
            parts.append(f"Instruksi: {system_prompt}")
        else:
            parts.append(f"Instruksi: {self.SYSTEM_PROMPTS['default']}")
        
        # Add context if provided
        if context:
            parts.append(f"\nKonteks:\n{context}")
        
        # Add user prompt
        parts.append(f"\nPertanyaan/Permintaan:\n{prompt}")
        
        return "\n".join(parts)
    
    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """Format retrieved contexts for the prompt."""
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("source", "Dokumen")
            content = ctx.get("content", ctx.get("text", ""))
            formatted.append(f"[{i}] Sumber: {source}\n{content}")
        
        return "\n\n".join(formatted)
    
    def _format_chat_history(self, history: List[ChatMessage]) -> str:
        """Format chat history for context."""
        formatted = []
        for msg in history[-5:]:  # Last 5 messages
            role = "Pengguna" if msg.role == "user" else "Asisten"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: ~4 characters per token for Indonesian/English
        total_chars = len(prompt) + len(response)
        return total_chars // 4


# Singleton instance
_llm_service: Optional[GeminiLLMService] = None


def get_llm_service() -> GeminiLLMService:
    """Get or create the LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = GeminiLLMService()
    return _llm_service
