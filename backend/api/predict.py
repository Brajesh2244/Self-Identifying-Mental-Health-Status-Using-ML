from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class MentalHealthRequest(BaseModel):
    phq9_score: int
    gad7_score: int
    family_history: bool
    sleep_disturbance: int

@router.post("/predict/mental-health")
async def predict_mental_health(request: MentalHealthRequest):
    # In Phase 3, this will invoke the Stacked Ensemble Model (.pkl)
    # For now, return the mock AI Insight Structure
    return {
        "status": "success",
        "risk_level": "Moderate Risk",
        "confidence_score": 0.88,
        "ai_insight": "Your overall psychological stability has improved by 42% since October.",
        "action_plan": [
            "10 minutes meditation",
            "30 minutes walking",
            "Sleep before 11 PM"
        ]
    }
