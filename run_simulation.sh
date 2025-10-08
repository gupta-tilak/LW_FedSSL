#!/bin/bash

# Kill any processes on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 2

# Run simulation and save output
python3 simulate_clients.py --num-clients 10 --mode lwfedssl > simulation_output.log 2>&1

echo "Simulation complete! Check simulation_output.log for full output"
echo ""
echo "=== SUMMARY ===="
tail -50 simulation_output.log
