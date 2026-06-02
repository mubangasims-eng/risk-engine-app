import numpy as np
import pandas as pd
from scipy.stats import poisson
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict
import json
import uuid
import asyncio
import sqlite3
import random
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# SYSTEM DATA STRUCTURES
# =====================================================================

@dataclass
class SyndicateUnifiedTick:
    event_id: str
    sport: str
    league: str
    match_id: str
    market: str
    selection: str
    bookmaker: str
    best_available_odds: float
    price_ladder: List[Dict[str, float]]
    timestamp: datetime
    bet_delay: int
    in_play_minute: int
    live_spatial_xt: float
    competition_tier: str = "PRO"
    live_total_goals: int = 0

class AsyncIngestionMesh:
    """Enterprise FIFO queue balancing raw multi-source vector ingestion"""
    def __init__(self, max_size: int = 500000):
        self.unified_queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.metrics = defaultdict(int)
        
    async def ingest_syndicate_tick(self, tick: SyndicateUnifiedTick):
        if tick.best_available_odds < 1.01: return
        try:
            self.unified_queue.put_nowait(tick)
            self.metrics['packets_routed'] += 1
        except asyncio.QueueFull:
            self.metrics['packets_dropped'] += 1

# =====================================================================
# SYSTEM ENVIRONMENT RISK PROFILER
# =====================================================================

class CompetitionRiskProfiler:
    """Evaluates systemic structural risk anomalies in erratic or lower-tier leagues"""
    @staticmethod
    def get_variance_modifiers(tier: str) -> Dict[str, Any]:
        if tier == "YOUTH_ACADEMY":
            return {
                "kelly_dampener": 0.25,
                "fatigue_acceleration": 1.85,
                "under_market_ban": True
            }
        elif tier == "REGIONAL_LOWER":
            return {
                "kelly_dampener": 0.50,
                "fatigue_acceleration": 1.60,
                "under_market_ban": False
            }
        return {
            "kelly_dampener": 1.00,
            "fatigue_acceleration": 1.45,
            "under_market_ban": False
        }

# =====================================================================
# SHARP VOL-WEIGHTED SENTIMENT CLUSTER
# =====================================================================

class MultiSharpSentimentCluster:
    """Aggregates price velocities across all sharp bookmakers simultaneously"""
    def __init__(self):
        self.sharp_weights = {
            "PINNACLE": 0.50,
            "BETFAIR_EXCHANGE": 0.35,
            "LMAX": 0.15
        }
        self.market_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    def register_tick_and_get_drift(self, tick: SyndicateUnifiedTick) -> float:
        if tick.bookmaker not in self.sharp_weights:
            return 0.0
            
        m_books = self.market_matrix[tick.match_id][tick.market][tick.selection]
        m_books[tick.bookmaker].append(tick.best_available_odds)
        m_books[tick.bookmaker] = m_books[tick.bookmaker][-5:]
        
        total_drift = 0.0
        for sharp_name, weight in self.sharp_weights.items():
            history = m_books.get(sharp_name, [])
            if len(history) >= 2:
                delta = history[-1] - history[-2]
                if delta < 0:
                    total_drift -= (abs(delta) * weight * 0.15)
                elif delta > 0:
                    total_drift += (delta * weight * 0.08)
                    
        return float(np.clip(total_drift, -0.12, 0.12))

# =====================================================================
# SPATIAL MODULING ENGINE
# =====================================================================

class SpatialTelemetryThreatEvaluator:
    """Converts 2D player/ball coordinates into predictive goal components"""
    def __init__(self):
        self.xt_matrix = np.random.uniform(0.01, 0.18, (12, 8))

    def evaluate_frame_telemetry(self, ball_x: float, ball_y: float, density_factor: float) -> float:
        grid_x = min(11, max(0, int(ball_x / 8.5)))
        grid_y = min(7, max(0, int(ball_y / 8.5)))
        base_threat = self.xt_matrix[grid_x, grid_y]
        return float(base_threat * (1.0 - (density_factor * 0.35)))

# =====================================================================
# NON-LINEAR WEIBULL STATE-SPACE MODEL
# =====================================================================

class WeibullInPlayStateSpaceModel:
    """Replaces linear time decay with a context-tuned non-linear Weibull hazard rate"""
    def __init__(self, baseline_home_weight: float = 1.12, scale_lambda: float = 0.012, shape_k: float = 1.45):
        self.base_home_weight = baseline_home_weight
        self.lmbda = scale_lambda  
        self.k = shape_k          

    def evaluate_live_parameters(self, minute: int, global_sentiment: float, spatial_xt: float, fatigue_accel: float) -> Tuple[float, float]:
        base_rem_pct = max(0.0, (90.0 - minute) / 90.0)
        
        effective_k = max(self.k, fatigue_accel)
        weibull_fatigue_modifier = np.exp(- (self.lmbda * minute) ** effective_k)
        effective_time_factor = (base_rem_pct * 0.4) + (weibull_fatigue_modifier * 0.6)
        
        mu_home = (1.65 * 0.90 * 1.35) * self.base_home_weight * effective_time_factor + (spatial_xt * 1.25) + global_sentiment
        mu_away = (1.30 * 1.10 * 1.35) * effective_time_factor
        
        return max(0.01, mu_home), max(0.01, mu_away)

# =====================================================================
# BIVARIATE POISSON CALCULATION DISTRIBUTIONS
# =====================================================================

class BivariatePoissonPredictor:
    def calculate_bivariate_probability(self, x: int, y: int, mu1: float, mu2: float, lambda3: float) -> float:
        prob = np.exp(-(mu1 + mu2 + lambda3)) * ((mu1**x) / np.math.factorial(x)) * ((mu2**y) / np.math.factorial(y))
        if lambda3 > 0 and x > 0 and y > 0:
            prob *= (1 + lambda3 * ((x / mu1) - 1) * ((y / mu2) - 1))
        return max(0.0, prob)

    def generate_true_probabilities(self, mu_home: float, mu_away: float, covariance: float = 0.05) -> Dict:
        max_goals = 6
        matrix = np.zeros((max_goals, max_goals))
        for i in range(max_goals):
            for j in range(max_goals):
                matrix[i, j] = self.calculate_bivariate_probability(i, j, mu_home, mu_away, covariance)
        
        matrix_sum = np.sum(matrix)
        if matrix_sum > 0: matrix /= matrix_sum

        return {
            "1X2": {
                "1": float(np.sum(np.tril(matrix, -1))), 
                "X": float(np.sum(np.diag(matrix))), 
                "2": float(np.sum(np.triu(matrix, 1)))
            },
            "OU_2.5": {
                "Over": float(1.0 - np.sum([matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i + j < 2.5])),
                "Under": float(np.sum([matrix[i, j] for i in range(max_goals) for j in range(max_goals) if i + j < 2.5]))
            }
        }

# =====================================================================
# EXECUTIONS & PORTFOLIO ALLOCATORS
# =====================================================================

class MarketLiquidityDepthFilter:
    @staticmethod
    def calculate_vwap_execution(price_ladder: List[Dict[str, float]], target_stake: float) -> Tuple[bool, float]:
        if not price_ladder:
            return False, 0.0
        accumulated_stake = 0.0
        weighted_price_sum = 0.0
        for layer in price_ladder:
            price, available = layer['price'], layer['available']
            fill = min(target_stake - accumulated_stake, available)
            weighted_price_sum += fill * price
            accumulated_stake += fill
            if accumulated_stake >= target_stake: break
        if accumulated_stake < target_stake: return False, 0.0
        return True, round(weighted_price_sum / target_stake, 4)

class InstitutionalKellyRiskManager:
    def __init__(self, total_bankroll: float, kelly_fraction: float = 0.20):
        self.bankroll = total_bankroll
        self.fraction = kelly_fraction

    def evaluate_risk_allocation(self, true_prob: float, vwap_odds: float) -> Tuple[bool, float]:
        if vwap_odds <= 1.0: return False, 0.0
        edge = true_prob - (1.0 / vwap_odds)
        if edge <= 0.008: return False, 0.0
        
        b = vwap_odds - 1.0
        optimal_f = (true_prob * b - (1.0 - true_prob)) / b
        return True, round(optimal_f * self.bankroll * self.fraction, 2)

class CrossMarketHedgingRing:
    @staticmethod
    def generate_correlated_hedges(market: str, selection: str, prime_stake: float) -> List[Dict]:
        dependency_map = {"1": [("OU_2.5", "Under")], "2": [("OU_2.5", "Under")]}
        if market != "1X2" or selection not in dependency_map: return []
        return [{
            "hedge_id": str(uuid.uuid4()), 
            "target_market": m, 
            "target_selection": s, 
            "allocated_hedge_stake": round(prime_stake * 0.30, 2)
        } for m, s in dependency_map[selection]]

# =====================================================================
# NON-BLOCKING ASYNC WRITE PERSISTENCE LAYER
# =====================================================================

class AsynchronousSignalStore:
    def __init__(self, db_path: str = "syndicate_enterprise.db"):
        self.db_path = db_path
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self._create_schema()
        self.worker_task = None

    def _create_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    signal_id TEXT PRIMARY KEY, match_id TEXT, market TEXT,
                    selection TEXT, vwap_odds REAL, true_probability REAL,
                    calculated_stake REAL, logged_at TEXT
                )
            """)
            conn.commit()

    async def start_worker(self):
        self.worker_task = asyncio.create_task(self._persistence_worker_loop())

    async def queue_signal_write(self, payload: Tuple):
        await self.write_queue.put(payload)

    async def _persistence_worker_loop(self):
        while True:
            try:
                payload = await self.write_queue.get()
                await asyncio.to_thread(self._execute_write, payload)
                self.write_queue.task_done()
            except asyncio.CancelledError:
                break

    def _execute_write(self, payload: Tuple):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO executions (
                    signal_id, match_id, market, selection, vwap_odds, true_probability, calculated_stake, logged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, payload)
            conn.commit()

    async def stop_worker(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

# =====================================================================
# CONSOLIDATED SYSTEM COORDINATION CORE
# =====================================================================

class EnterpriseSyndicateEngine:
    def __init__(self, bankroll: float, db_path: str = "syndicate_enterprise.db"):
        self.mesh = AsyncIngestionMesh()
        self.db = AsynchronousSignalStore(db_path=db_path)
        self.sentiment_analyzer = MultiSharpSentimentCluster()
        self.spatial_evaluator = SpatialTelemetryThreatEvaluator()
        self.state_space = WeibullInPlayStateSpaceModel()
        self.predictor = BivariatePoissonPredictor()
        self.risk_manager = InstitutionalKellyRiskManager(total_bankroll=bankroll)
        self.running = False
        self.telemetry = {"active_positions": 0}
        self.execution_history = []

    async def start_processing_loop(self):
        self.running = True
        await self.db.start_worker()
        asyncio.create_task(self._unified_execution_worker())

    async def stop_processing_loop(self):
        self.running = False
        await self.db.stop_worker()

    async def _unified_execution_worker(self):
        while self.running:
            try:
                tick: SyndicateUnifiedTick = await asyncio.wait_for(self.mesh.unified_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            
            try:
                risk = CompetitionRiskProfiler.get_variance_modifiers(tick.competition_tier)
                
                if risk["under_market_ban"] and ("Under" in tick.selection or "Under" in tick.market):
                    print(f"RISK BLOCK | Terminating execution track on {tick.match_id} [{tick.market}:{tick.selection}]")
                    self.mesh.unified_queue.task_done()
                    continue

                if tick.live_total_goals >= 2 and tick.in_play_minute <= 25 and ("Under" in tick.selection or "Under" in tick.market):
                    print(f"VELOCITY OVERRIDE | Match {tick.match_id} exceeded macro-volatility parameters.")
                    self.mesh.unified_queue.task_done()
                    continue
                
                global_drift = self.sentiment_analyzer.register_tick_and_get_drift(tick)
                
                mu_h, mu_a = self.state_space.evaluate_live_parameters(
                    minute=tick.in_play_minute, global_sentiment=global_drift, 
                    spatial_xt=tick.live_spatial_xt, fatigue_accel=risk["fatigue_acceleration"]
                )
                
                true_probs = self.predictor.generate_true_probabilities(mu_h, mu_a)
                
                if tick.market in true_probs and tick.selection in true_probs[tick.market]:
                    target_true_prob = true_probs[tick.market][tick.selection]
                else:
                    target_true_prob = 1.0 / tick.best_available_odds
                    
                has_edge, test_stake = self.risk_manager.evaluate_risk_allocation(target_true_prob, tick.best_available_odds)
                
                if has_edge and test_stake > 0:
                    liquid, vwap_price = MarketLiquidityDepthFilter.calculate_vwap_execution(tick.price_ladder, test_stake)
                    if liquid:
                        final_execution, final_stake = self.risk_manager.evaluate_risk_allocation(target_true_prob, vwap_price)
                        if final_execution and final_stake > 0:
                            allocated_stake = round(final_stake * risk["kelly_dampener"], 2)
                            if allocated_stake > 0:
                                self.telemetry["active_positions"] += 1
                                hedge_positions = CrossMarketHedgingRing.generate_correlated_hedges(tick.market, tick.selection, allocated_stake)
                                
                                execution_record = {
                                    "signal_id": str(uuid.uuid4()),
                                    "match_id": tick.match_id,
                                    "market": tick.market,
                                    "selection": tick.selection,
                                    "vwap_odds": vwap_price,
                                    "true_probability": target_true_prob,
                                    "allocated_stake": allocated_stake,
                                    "competition_tier": tick.competition_tier,
                                    "hedges": hedge_positions
                                }
                                self.execution_history.append(execution_record)
                                
                                print(f"SYSTEM EXECUTED | Match: {tick.match_id} ({tick.competition_tier}) | Sizing: ${allocated_stake}")
                                
                                db_payload = (
                                    execution_record["signal_id"],
                                    tick.match_id,
                                    tick.market,
                                    tick.selection,
                                    vwap_price,
                                    target_true_prob,
                                    allocated_stake,
                                    datetime.now().isoformat()
                                )
                                await self.db.queue_signal_write(db_payload)
                        
            except Exception as e:
                print(f"Error processing tick: {e}")
            finally:
                self.mesh.unified_queue.task_done()

    async def inject_live_tick(self, raw_x: float, raw_y: float, dense: float, book: str, odds: float, 
                               min_val: int = 44, tier: str = "PRO", live_goals: int = 0, 
                               market: str = "1X2", selection: str = "1"):
        m_id = "MCI_RMA_2026"
        threat = self.spatial_evaluator.evaluate_frame_telemetry(raw_x, raw_y, dense)
        mock_ladder = [{'price': odds, 'available': 25000.0}, {'price': odds - 0.02, 'available': 50000.0}]
        
        tick = SyndicateUnifiedTick(
            event_id="E_UCL_99", sport="soccer", league="UCL", match_id=m_id, market=market, selection=selection,
            bookmaker=book, best_available_odds=odds, price_ladder=mock_ladder, timestamp=datetime.now(),
            bet_delay=0, in_play_minute=min_val, live_spatial_xt=threat, competition_tier=tier, live_total_goals=live_goals
        )
        await self.mesh.ingest_syndicate_tick(tick)

    def get_telemetry(self) -> Dict:
        return {
            "active_positions": self.telemetry["active_positions"],
            "packets_routed": self.mesh.metrics['packets_routed'],
            "packets_dropped": self.mesh.metrics['packets_dropped'],
            "total_executions": len(self.execution_history)
        }