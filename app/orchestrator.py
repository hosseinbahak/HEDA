import time
from typing import Dict, Any, Tuple, Optional
from app.agents.prosecutor import Prosecutor
from app.agents.defender import Defender
from app.agents.judge import Judge
from app.agents.jury import Jury
from app.graph.round_table_graph import RoundTableGraph
from app.utils.logger import get_logger

class Orchestrator:
    """Orchestrates the evaluation process between different modes."""
    
    def __init__(self, provider, mode="traditional", max_rounds=2, verbose=False):
        self.provider = provider
        self.mode = mode
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.structured_logger = get_logger(__name__)
        
        # Initialize components based on mode
        if mode == "roundtable":
            self.round_table = RoundTableGraph(provider, max_rounds=max_rounds)
            # Also initialize traditional components as fallback
            self._init_traditional_components()
        else:
            self._init_traditional_components()
    
    def _init_traditional_components(self):
        """Initialize traditional mode components."""
        self.prosecutor = Prosecutor(self.provider)
        self.defender = Defender(self.provider)
        self.judge = Judge(self.provider)
        self.jury = Jury(self.provider)
    
    def evaluate_text(self, text: str) -> Dict[str, Any]:
        """Main evaluation entry point."""
        self.structured_logger.info("eval_start")
        
        if self.mode == "roundtable":
            return self._evaluate_with_roundtable(text)
        else:
            return self._evaluate_traditional(text)
    
    def _evaluate_with_roundtable(self, text: str) -> Dict[str, Any]:
        """Evaluate using roundtable discussion mode."""
        try:
            result = self.round_table.evaluate_text(text)
            return result
        except Exception as e:
            self.structured_logger.error("roundtable_error", error=str(e))
            self.structured_logger.info("falling_back_to_traditional")
            return self._evaluate_traditional(text)
    
    def _evaluate_traditional(self, text: str) -> Dict[str, Any]:
        """Traditional evaluation mode with separate components."""
        # Ensure traditional components are initialized
        if not hasattr(self, 'prosecutor'):
            self._init_traditional_components()
        
        start = time.time()
        
        # Prosecutor phase
        p, p_ms = self._time_call(self.prosecutor.run_on, text, "")
        
        # Defender phase
        d, d_ms = self._time_call(self.defender.run_on, text, p)
        
        # Judge phase
        j, j_ms = self._time_call(self.judge.run_on, text, p, d)
        
        # Jury phase
        jury_result, jury_ms = self._time_call(self.jury.run_on, text, p, d, j)
        
        total_time = time.time() - start
        
        return {
            "prosecutor": p,
            "defender": d,
            "judge": j,
            "jury": jury_result,
            "mode": "traditional",
            "timings": {
                "prosecutor_ms": p_ms,
                "defender_ms": d_ms,
                "judge_ms": j_ms,
                "jury_ms": jury_ms,
                "total_ms": total_time * 1000
            }
        }
    
    def _time_call(self, func, *args) -> Tuple[Any, float]:
        """Time a function call and return result with timing."""
        start = time.time()
        result = func(*args)
        elapsed = (time.time() - start) * 1000
        return result, elapsed