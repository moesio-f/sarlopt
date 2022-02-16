import os

SRC_DIR = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.normpath(os.path.join(SRC_DIR, '..'))
OUTPUT_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'output'))
POLICIES_DIR = os.path.normpath(os.path.join(OUTPUT_DIR, 'policies'))
EXPERIMENTS_DIR = os.path.normpath(os.path.join(ROOT_DIR, 'experiments'))
BASELINES_DIR = os.path.normpath(os.path.join(EXPERIMENTS_DIR, 'baselines'))
TRAINING_DIR = os.path.normpath(os.path.join(EXPERIMENTS_DIR, 'training'))
