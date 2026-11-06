#!/bin/zsh
# Empreinte mémoire réelle (footprint, inclut la mémoire MPS) des processus realmlp/run.py, toutes les 30 s,
# tant que le PID $1 (la file d'attente) existe.
while kill -0 $1 2>/dev/null; do
  for p in $(pgrep -f "python3 realmlp/run.py"); do
    echo "$(date +%T) $(footprint $p 2>/dev/null | grep -o 'Footprint: [0-9]* [MG]B') $(ps -o command= -p $p | grep -o -- '--name [^ ]*')"
  done
  sleep 30
done
