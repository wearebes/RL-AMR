# RL-AMR
```bash
conda activate pinn
python -m train.train --config train/training.yaml

python -m evaluation.evaluate bumps
python -m evaluation.plot data/static_bump/degree.npz

python -m evaluation.evaluate advection
python -m evaluation.plot data/advection_IPN/degree.npz

python -m evaluation.evaluate transfer
python -m evaluation.plot data/static_bump/transfer.npz

python -m evaluation.plot data/advection_IPN/history.csv
python -m evaluation.evaluate bumps --config /path/to/evaluation.yaml
```
## Mesh smoothing/optimization
1. Create the Python environment:
```bash
conda create -n fsi python=3.12.0
conda activate fsi
```
2.for NN and Kratos-FSI2, mesquite environment packages
```bash
python -m pip install torch==2.14.0+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy==2.5.3


python -m pip install --only-binary=:all: \
  KratosMultiphysics==10.4.4 \
  KratosFluidDynamicsApplication==10.4.4 \
  KratosStructuralMechanicsApplication==10.4.4 \
  KratosFSIApplication==10.4.4 \
  KratosMeshMovingApplication==10.4.4 \
  KratosConstitutiveLawsApplication==10.4.4 \
  KratosMappingApplication==10.4.4 \
  KratosLinearSolversApplication==10.4.4

# the mesquite-2.3.0 build, first cd the main mesquite then build 
cd /mnt/e/Research/RL-AMR/mesquite 
mkdir -p build-linux 
cd build-linux 

CONFIG_SHELL=/bin/bash SHELL=/bin/bash \
bash ../mesquite-2.3.0/configure \
  --prefix="$HOME/.local/mesquite/2.3.0" \
  --without-cppunit \
  CC=gcc CXX=g++ CXXFLAGS='-O2 -std=c++98 -fPIC' &&

make SHELL=/bin/bash -j4 
make SHELL=/bin/bash install
```

# Build the native FSI2 bridge on the target Linux machine, from the repository root:

```bash
cmake -S Smoothing/fsi_turek_FSI2 -B Smoothing/fsi_turek_FSI2/build-mesquite \
  -DMESQUITE_ROOT="$HOME/.local/mesquite/2.3.0"
cmake --build Smoothing/fsi_turek_FSI2/build-mesquite -j4
```

3. Run the four formal FSI2 cases (20 s each), from the repository root:

```bash
# Local Linux / WSL: selects the fsi Conda environment automatically.
bash job.sh

# Slurm HPC: uses the CPU allocation declared at the top of job.sh.
sbatch job.sh

# Optional: different environment, thread count, duration and output directory.
CONDA_ENV=pinn OMP_NUM_THREADS=32 bash job.sh 0.05 Smoothing/data/new_smoke
# If Conda is not on PATH, set CONDA_EXE=/path/to/conda.
# Alternatively, PYTHON=/path/to/env/bin/python bash job.sh
```

Local runs inherit `OMP_NUM_THREADS` if set; otherwise Kratos uses its runtime
default. Slurm runs use `SLURM_CPUS_PER_TASK`. The four cases run sequentially.
The target machine must already have the environment and native bridge installed.
