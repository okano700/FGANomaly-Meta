#PBS -l walltime=1:00:00 
#PBS -l select=1:ngpus=1
#PBS -N VAIBRASIL
#PBS -oe
#PBS -m abe
#PBS -M emerson.okano@unifesp.br

  
#PBS -V
python ~/FGANomaly-Meta/runFGAN.py --path ~/datsets/UCR_Anomaly_FullData/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt --WL 1000 --n 5 --ds UCR --i 0
