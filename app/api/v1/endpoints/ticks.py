from fastapi import APIRouter, HTTPException, Depends
from typing import List
from app.models.schemas import TickInjectionRequest, ExecutionResponse
from app.services.syndicate import get_engine

router = APIRouter(prefix="/ticks", tags=["ticks"])

@router.post("/inject", response_model=dict)
async def inject_tick(request: TickInjectionRequest, engine = Depends(get_engine)):
    """Inject a live market tick into the processing pipeline"""
    try:
        await engine.inject_live_tick(
            raw_x=request.raw_x,
            raw_y=request.raw_y,
            dense=request.density_factor,
            book=request.bookmaker,
            odds=request.odds,
            min_val=request.minute,
            tier=request.tier,
            live_goals=request.live_goals,
            market=request.market,
            selection=request.selection
        )
        return {"status": "success", "message": "Tick injected into pipeline"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[ExecutionResponse])
async def get_execution_history(engine = Depends(get_engine)):
    """Get recent execution history"""
    return [ExecutionResponse(**exec) for exec in engine.execution_history[-100:]]