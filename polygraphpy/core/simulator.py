from abc import ABC, abstractmethod
from typing import Any, Dict

class Simulator(ABC):
    """Abstract base class for simulation workflows."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
    
    @abstractmethod
    def prepare_input(self, input_data: Any) -> None:
        """Prepare input files or data for the simulation."""
        pass
    
    @abstractmethod
    def run(self) -> None:
        """Execute the simulation."""
        pass
    
    @abstractmethod
    def process_output(self) -> Any:
        """Process and return simulation results."""
        pass