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
1.may create the python env, may larger than version 3.12(3.13, 3.14)
```bash
conda create -n fsi python==3.12
conda activate fsi
```
2.for NN and Kratos-FSI2, mesquite environment packages
```bash
pip install torch
pip install numpy


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