#!/bin/bash
#SBATCH --job-name=RF_Astrai
#SBATCH --output=logs_rf_%j.out
#SBATCH --error=logs_rf_%j.err
#SBATCH --partition=leonardo_sys_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32      # Numero di core da usare (RF ne beneficerà molto)
#SBATCH --time=02:00:00         # Tempo massimo stimato
#SBATCH --account=<TUO_ACCOUNT> # Inserisci il tuo account Leonardo

# Carica l'ambiente Python/Conda
module load python
source activate tuo_ambiente_conda

# Esegui il codice
python random_forest_leonardo.py
