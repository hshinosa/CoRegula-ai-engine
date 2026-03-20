import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

# Add root
sys.path.append(os.getcwd())

from app.services.orchestration import get_orchestrator
from app.services.mongodb_logger import get_mongo_logger
from app.services.notification_service import get_notification_service

async def run_proactive_integration_test():
    """
    Comprehensive test for all Kolabri proactive features.
    """
    print("=" * 70)
    print(" KOLABRI PROACTIVE SYSTEM INTEGRATION TEST")
    print("=" * 70)

    # 0. Initialization
    mongo_logger = get_mongo_logger()
    await mongo_logger.connect()
    
    # Mock notification to avoid real API calls
    notification_service = get_notification_service()
    notification_service.notify_teacher = AsyncMock(return_value=True)
    notification_service.send_intervention = AsyncMock(return_value=True)
    
    orchestrator = get_orchestrator()
    group_id = f"test-group-{int(asyncio.get_event_loop().time())}"
    user_id = f"student-1"

    # 1. Test Goal Validation & Socratic Hinting
    print("\n[PHASE 1] Goal Setting & Socratic Hinting")
    # Intentional weak goal
    res = await orchestrator.validate_goal("Belajar Python", user_id, group_id)
    print(f"    Goal: 'Belajar Python'")
    print(f"    Socratic Hint: {res.get('socratic_hint')}")
    if res.get('socratic_hint'):
        print("    [PASS] Socratic hint generated for weak goal.")
    else:
        print("    [FAIL] No hint generated.")
    
    # 2. Test Dashboard Advice
    print("\n[PHASE 2] Teacher Advisory Engine")
    # We already sent messages in phase 2 above (merged)
    group_dash = await orchestrator.get_group_dashboard_data(group_id)
    print(f"    Group Advice: {group_dash.get('teacher_advice')}")
    if group_dash.get('teacher_advice'):
        print("    [PASS] Teacher advice narrative found.")
    
    indiv_dash = await orchestrator.get_individual_dashboard_data(user_id)
    print(f"    Individual Advice: {indiv_dash.get('personal_advice')}")
    if indiv_dash.get('personal_advice'):
        print("    [PASS] Individual advice narrative found.")

    # 2. Test Message Handling & Anomaly Detection
    print("\n[PHASE 2] Discussion & Anomaly Detection")
    # Simulate direct messages to trigger sequence anomaly (skipped goals in some contexts)
    # or just normal flow
    messages = [
        "Bagaimana cara kerja rekursi?",
        "Saya mencoba mengimplementasikan Fibonacci.",
        "Oke saya paham."
    ]
    for msg in messages:
        await orchestrator.handle_message(user_id, group_id, msg, topic="Algorithms")
    
    print("  - Messages logged and analyzed.")

    # 3. Test Dashboard APIs (The new split dashboard)
    print("\n[PHASE 3] Teacher Dashboards (Group vs Individual)")
    
    print("  - Fetching Group Dashboard...")
    group_dash = await orchestrator.get_group_dashboard_data(group_id)
    print(f"    Context: {group_dash.get('context')}")
    print(f"    Metrics: {group_dash.get('metrics')}")
    
    print("  - Fetching Individual Dashboard...")
    indiv_dash = await orchestrator.get_individual_dashboard_data(user_id)
    print(f"    User: {indiv_dash.get('user_id')}")
    print(f"    Personal HOT: {indiv_dash.get('personal_metrics', {}).get('hot_percentage')}%")

    # 4. Final Verification
    print("\n" + "=" * 70)
    print(" INTEGRATION TEST COMPLETE")
    print("=" * 70)
    
    await mongo_logger.close()

if __name__ == "__main__":
    asyncio.run(run_proactive_integration_test())
