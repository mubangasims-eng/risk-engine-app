from fastapi import APIRouter, Depends
from app.models.schemas import TelemetryResponse
from app.services.syndicate import get_engine

router = APIRouter(prefix="/positions", tags=["positions"])

@router.get("/telemetry", response_model=TelemetryResponse)
async def get_telemetry(engine = Depends(get_engine)):
    """Get real-time system telemetry"""
    return TelemetryResponse(**engine.get_telemetry())

@router.get("/active-count")
async def get_active_positions_count(engine = Depends(get_engine)):
    """Get count of active positions"""
    return {"active_positions": engine.telemetry["active_positions"]}