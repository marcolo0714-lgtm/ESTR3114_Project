import sys
from pathlib import Path
import torch
torch.set_num_threads(1)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
