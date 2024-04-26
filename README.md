# ATLAS_experiment

This project focuses on studying the HIGGS BOSON using the full LHC dataset recorded at 13 TeV proton-proton collision energy (HZZ Analysis)
The idea is decay the process from Higgs particle (H) to 4 leptons (llll) using docker containers. 

### To run the scrpit 
#### Worker script
docker run -d --name hzz-worker worker-image python hzz-worker.py
#### Counter script
docker run -d --name hzz-counter counter-image python hzz-counter.py
#### Collector script
docker run -d --name hzz-collector collector-image python hzz-collector.py

