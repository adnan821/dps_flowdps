import sys
from pathlib import Path

# Make `from src...` imports work when running pytest from the project root.
sys.path.insert(0, str(Path(__file__).parent))
