#SBATCH --job-name fr1996((60,80),(80,100))
#SBATCH --error=/your_path/fr1996((60,80),(80,100)).err
#SBATCH --output=/your_path/fr1996((60,80),(80,100)).out
#SBATCH --time=7-00:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 24
#SBATCH --partition=long
#SBATCH --mem=4G
date
python GS_detectFluxRope_multiprocessing.py 1996 ((60,80),(80,100))
date
