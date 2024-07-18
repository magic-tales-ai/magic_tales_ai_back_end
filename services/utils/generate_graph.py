import sys
from pyan import create_callgraph

filename = '/media/neoadmin/Treasure/0000NC-laptop/personal/00-MAGIC-TALES/magit_tales_pro/magic_tales_ai_back_end/services/orchestrator/orchestrator.py'
graph = create_callgraph([filename], format='dot', grouped=True, annotated=True)

with open('output.dot', 'w') as f:
    f.write(graph)