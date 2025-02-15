import os
import sys

# UPDATE SEARCH PATH
def get_base_path():
    cwd = os.path.basename(os.getcwd())
    if cwd.lower() != 'power-grid-time-series-anomaly-detection':
        return os.path.dirname(os.getcwd())
    else:
        return os.getcwd()

root = get_base_path()

for base, dirs, files in os.walk(root):
    if not any(i in base for i in ['env', '.git']):
        sys.path.append(base)