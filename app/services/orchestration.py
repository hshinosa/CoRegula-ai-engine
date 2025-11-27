"""
Orchestration Service - Teacher-AI Complementarity
CoRegula AI Engine

Implements the Teacher-AI Complementarity principle where AI acts as an assistant
that monitors in real-time and provides notifications/interventions (scaffolding)
to instructors/groups only when needed, keeping the instructor's role central.

Combines:
- RAG Engine for knowledge-based responses
- NLP Analytics for engagement measurement
- Process Mining Logger for research data collection
- Intervention triggers based on discussion quality

Reference: LUMILO framework for intelligent mediation support
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from app.core.logging import get_logger
from app.core.config import settings
from app.services.rag import RAGPipeline, get_rag_pipeline, RAGResult
from app.services.nlp_analytics import (
    EngagementAnalyzer, 
    get_engagement_analyzer,
    EngagementAnalysis,
    EngagementType
)
from app.services.intervention import (
    ChatInterventionService,
    get_intervention_service,
    InterventionResult,
    InterventionType
)
from app.utils.logger import (
    ProcessMiningLogger,
    get_process_mining_logger,
    ActivityType,
    Lifecycle
)

logger = get_logger(__name__)


@dataclass
class OrchestrationResult:
    """Result from orchestration processing."""
    reply: str                          # Bot response
    intervention: Optional[str]         # System intervention message (if triggered)
    intervention_type: Optional[str]    # Type of intervention
    analytics: Dict[str, Any]           # Engagement analytics
    action_taken: str                   # FETCH or NO_FETCH
    should_notify_teacher: bool         # Whether to alert instructor
    quality_score: Optional[float]      # Discussion quality score
    success: bool
    error: Optional[str] = None


class Orchestrator:
    """
    Central orchestration service for Teacher-AI Complementarity.
    
    Features:
    - Real-time engagement monitoring
    - Policy-based RAG responses
    - Automatic intervention triggering
    - Process mining event logging
    - Teacher notification system
    
    Workflow:
    1. Receive student message
    2. Analyze engagement (NLP Analytics)
    3. Generate RAG response with policy optimization
    4. Log events for process mining
    5. Check intervention triggers
    6. Return response with optional intervention
    
    Configuration (from settings):
    - NLP_LOW_LEXICAL_THRESHOLD: Trigger intervention below this lexical variety
    - NLP_QUALITY_ALERT_THRESHOLD: Notify teacher below this quality score
    - INTERVENTION_COOLDOWN_MINUTES: Minimum time between interventions
    - INTERVENTION_MIN_MESSAGES: Minimum messages before quality check
    """
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        analyzer: Optional[EngagementAnalyzer] = None,
        intervention_service: Optional[ChatInterventionService] = None,
        pm_logger: Optional[ProcessMiningLogger] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            rag_pipeline: RAG pipeline for responses
            analyzer: Engagement analyzer for NLP
            intervention_service: Intervention generator
            pm_logger: Process mining logger
        """
        self.rag = rag_pipeline or get_rag_pipeline()
        self.analyzer = analyzer or get_engagement_analyzer()
        self.intervention = intervention_service or get_intervention_service()
        self.pm_logger = pm_logger or get_process_mining_logger()
        
        # Track intervention history per group
        self._last_intervention: Dict[str, datetime] = {}
        self._group_messages: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("orchestrator_initialized")
    
    async def handle_message(
        self,
        user_id: str,
        group_id: str,
        message: str,
        topic: Optional[str] = None,
        collection_name: Optional[str] = None,
        course_id: Optional[str] = None,
        chat_room_id: Optional[str] = None
    ) -> OrchestrationResult:
        """
        Handle an incoming message through the full orchestration pipeline.
        
        Args:
            user_id: ID of the user sending the message
            group_id: ID of the group/chat room
            message: The message content
            topic: Optional discussion topic for context
            collection_name: Optional vector store collection
            course_id: Optional course ID for context
            chat_room_id: Optional chat room ID for context
            
        Returns:
            OrchestrationResult with response, analytics, and intervention
        """
        # Extract course_id from collection_name if not provided
        if not course_id and collection_name and collection_name.startswith("course_"):
            course_id = collection_name.replace("course_", "")
        
        try:
            # 1. Analyze Engagement & NLP
            analytics = self.analyzer.analyze_interaction(message)
            analytics_dict = self._analytics_to_dict(analytics)
            
            logger.info(
                "message_analyzed",
                group_id=group_id,
                user_id=user_id,
                course_id=course_id,
                engagement=analytics.engagement_type.value,
                lexical_variety=analytics.lexical_variety,
                is_hot=analytics.is_higher_order
            )
            
            # 2. RAG Response Generation with Policy
            rag_result = await self.rag.query(
                query=message,
                collection_name=collection_name
            )
            bot_reply = rag_result.answer if rag_result.success else "Maaf, terjadi kesalahan dalam memproses pertanyaan."
            
            # Determine action (FETCH/NO_FETCH) from pipeline behavior
            action_taken = "FETCH" if rag_result.sources else "NO_FETCH"
            
            # 3. Log Events for Process Mining (with full context)
            # Log student message
            self.pm_logger.log_student_message(
                group_id=group_id,
                user_id=user_id,
                message_length=len(message),
                course_id=course_id,
                chat_room_id=chat_room_id,
                topic=topic,
                engagement_type=analytics.engagement_type.value,
                lexical_variety=analytics.lexical_variety,
                is_hot=analytics.is_higher_order
            )
            
            # Log bot response
            self.pm_logger.log_bot_response(
                group_id=group_id,
                action_taken=action_taken,
                response_length=len(bot_reply),
                course_id=course_id,
                chat_room_id=chat_room_id,
                topic=topic
            )
            
            # 4. Track message history for group
            self._track_message(group_id, user_id, message, analytics)
            
            # 5. Check Intervention Triggers
            intervention_msg = None
            intervention_type = None
            should_notify_teacher = False
            quality_score = None
            
            # Get group quality metrics
            group_messages = self._group_messages.get(group_id, [])
            if len(group_messages) >= settings.INTERVENTION_MIN_MESSAGES:
                quality_result = self.analyzer.get_discussion_quality_score(
                    [m["message"] for m in group_messages[-10:]]
                )
                quality_score = quality_result["quality_score"]
                
                # Check if intervention needed
                intervention_needed, reason = self._should_intervene(
                    group_id=group_id,
                    analytics=analytics,
                    quality_score=quality_score
                )
                
                if intervention_needed:
                    intervention_msg = self._generate_intervention_message(
                        analytics=analytics,
                        quality_score=quality_score,
                        reason=reason,
                        topic=topic
                    )
                    intervention_type = reason
                    
                    # Log intervention with full context
                    self.pm_logger.log_intervention(
                        group_id=group_id,
                        intervention_type=reason,
                        trigger_reason=f"quality:{quality_score}, lexical:{analytics.lexical_variety}",
                        course_id=course_id,
                        chat_room_id=chat_room_id,
                        topic=topic
                    )
                    
                    # Update last intervention time
                    self._last_intervention[group_id] = datetime.now()
                
                # Check if teacher should be notified
                if settings.NOTIFY_TEACHER_ON_LOW_QUALITY and quality_score < settings.NLP_QUALITY_ALERT_THRESHOLD:
                    should_notify_teacher = True
            
            return OrchestrationResult(
                reply=bot_reply,
                intervention=intervention_msg,
                intervention_type=intervention_type,
                analytics=analytics_dict,
                action_taken=action_taken,
                should_notify_teacher=should_notify_teacher,
                quality_score=quality_score,
                success=True
            )
            
        except Exception as e:
            logger.error(
                "orchestration_failed",
                error=str(e),
                group_id=group_id,
                user_id=user_id
            )
            
            return OrchestrationResult(
                reply="Maaf, terjadi kesalahan dalam sistem.",
                intervention=None,
                intervention_type=None,
                analytics={},
                action_taken="ERROR",
                should_notify_teacher=False,
                quality_score=None,
                success=False,
                error=str(e)
            )
    
    async def get_group_analytics(self, group_id: str) -> Dict[str, Any]:
        """
        Get aggregated analytics for a group.
        
        Args:
            group_id: The group ID
            
        Returns:
            Dictionary with group-level analytics
        """
        messages = self._group_messages.get(group_id, [])
        
        if not messages:
            return {
                "group_id": group_id,
                "message_count": 0,
                "quality_score": None,
                "quality_breakdown": {
                    "lexical_score": 0,
                    "hot_score": 0,
                    "cognitive_ratio": 0,
                    "lexical_variety": 0,
                    "hot_percentage": 0,
                    "participation": 0
                },
                "participants": [],
                "participant_count": 0,
                "engagement_distribution": {},
                "hot_percentage": 0,
                "recommendation": None
            }
        
        # Get quality score
        quality = self.analyzer.get_discussion_quality_score(
            [m["message"] for m in messages]
        )
        
        # Get unique participants
        participants = list(set(m["user_id"] for m in messages))
        
        # Engagement distribution
        engagement_dist = {}
        for m in messages:
            eng_type = m.get("engagement_type", "behavioral")
            engagement_dist[eng_type] = engagement_dist.get(eng_type, 0) + 1
        
        # Calculate HOT percentage from messages
        hot_count = sum(1 for m in messages if m.get("is_hot", False))
        hot_percentage = (hot_count / len(messages)) * 100 if messages else 0
        
        # Get average lexical variety
        avg_lexical = sum(m.get("lexical_variety", 0) for m in messages) / len(messages) if messages else 0
        
        return {
            "group_id": group_id,
            "message_count": len(messages),
            "quality_score": quality["quality_score"],
            "quality_breakdown": {
                "lexical_score": quality["lexical_score"],
                "hot_score": quality["hot_score"],
                "cognitive_ratio": quality["cognitive_ratio"],
                "lexical_variety": round(avg_lexical, 3),
                "hot_percentage": round(hot_percentage, 1),
                "participation": len(participants)
            },
            "recommendation": quality["recommendation"],
            "participants": participants,
            "participant_count": len(participants),
            "engagement_distribution": engagement_dist,
            "hot_percentage": round(hot_percentage, 1)
        }
    
    def _should_intervene(
        self,
        group_id: str,
        analytics: EngagementAnalysis,
        quality_score: float
    ) -> tuple[bool, Optional[str]]:
        """
        Decide whether to trigger an intervention.
        
        Uses configurable thresholds from settings:
        - INTERVENTION_COOLDOWN_MINUTES
        - NLP_LOW_LEXICAL_THRESHOLD
        - NLP_QUALITY_ALERT_THRESHOLD
        
        Returns:
            Tuple of (should_intervene, reason)
        """
        # Check cooldown
        last_intervention = self._last_intervention.get(group_id)
        if last_intervention:
            minutes_since = (datetime.now() - last_intervention).total_seconds() / 60
            if minutes_since < settings.INTERVENTION_COOLDOWN_MINUTES:
                return False, None
        
        # Check low lexical variety (shallow discussion)
        if analytics.lexical_variety < settings.NLP_LOW_LEXICAL_THRESHOLD:
            return True, "low_lexical_variety"
        
        # Check low quality score
        if quality_score and quality_score < settings.NLP_QUALITY_ALERT_THRESHOLD:
            return True, "low_quality"
        
        return False, None
    
    def _generate_intervention_message(
        self,
        analytics: EngagementAnalysis,
        quality_score: Optional[float],
        reason: str,
        topic: Optional[str] = None
    ) -> str:
        """Generate an appropriate intervention message."""
        
        if reason == "low_lexical_variety":
            # Prompt for deeper thinking (HOT trigger)
            return (
                "💡 [Saran Bot: Coba jelaskan 'mengapa' kalian memilih opsi ini? "
                "Apa alasan di balik pendapat tersebut?]"
            )
        
        elif reason == "low_quality":
            return (
                "📚 [Saran Bot: Mari kita tingkatkan diskusi dengan menganalisis "
                "lebih dalam. Apa hubungan antara konsep yang sedang dibahas "
                f"{'dengan topik ' + topic if topic else ''}?]"
            )
        
        return (
            "💭 [Saran Bot: Jangan lupa untuk mendukung argumen kalian "
            "dengan alasan yang jelas!]"
        )
    
    def _track_message(
        self,
        group_id: str,
        user_id: str,
        message: str,
        analytics: EngagementAnalysis
    ):
        """Track message in group history."""
        if group_id not in self._group_messages:
            self._group_messages[group_id] = []
        
        self._group_messages[group_id].append({
            "user_id": user_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "engagement_type": analytics.engagement_type.value,
            "is_hot": analytics.is_higher_order,
            "lexical_variety": analytics.lexical_variety
        })
        
        # Keep only last 100 messages per group
        if len(self._group_messages[group_id]) > 100:
            self._group_messages[group_id] = self._group_messages[group_id][-100:]
    
    def _analytics_to_dict(self, analytics: EngagementAnalysis) -> Dict[str, Any]:
        """Convert EngagementAnalysis to dictionary."""
        return {
            "lexical_variety": analytics.lexical_variety,
            "engagement_type": analytics.engagement_type.value,
            "is_higher_order": analytics.is_higher_order,
            "hot_indicators": analytics.hot_indicators,
            "word_count": analytics.word_count,
            "unique_words": analytics.unique_words,
            "confidence": analytics.confidence
        }


# Singleton instance
_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
