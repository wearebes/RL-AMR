conda activate fsi

cd /Users/jcy/research/RL-AMR/mesquite/build

../mesquite-2.3.0/configure \
  --prefix=/Users/jcy/research/RL-AMR/mesquite/install \
  --without-cppunit \
  CC=/usr/bin/clang \
  CXX=/usr/bin/clang++ \
  CXXFLAGS="-O2 -std=c++98 -fPIC"

make -j4
make install