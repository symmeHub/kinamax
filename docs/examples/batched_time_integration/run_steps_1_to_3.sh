set -euo pipefail

echo "Running step01_integrate_until_convergence.py"
python step01_integrate_until_convergence.py

echo "Running step02_detect_orbits.py"
python step02_detect_orbits.py

echo "Running step03_calculate_orbits.py"
python step03_calculate_orbits.py

echo "Finished running steps 1 to 3."
