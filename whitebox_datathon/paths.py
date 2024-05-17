from pathlib import Path
import os

# current directory
SOURCE = Path(os.getcwd())

ROOT = SOURCE.parent

# data directory
DATA = ROOT / 'data'
DATA_RAW = DATA / 'raw'
DATA_PROCESSED = DATA / 'processed'
