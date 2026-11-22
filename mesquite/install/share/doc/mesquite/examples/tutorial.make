prefix = /Users/jcy/research/RL-AMR/mesquite/install
exec_prefix = ${prefix}
libdir = ${exec_prefix}/lib
includedir = ${prefix}/include

CXXFLAGS = -g
LIBS = -lmesquite
LDFLAGS = -L$(libdir)
CPPFLAGS = -I$(includedir)

tutorial: tutorial.cpp
	$(CXX) -o $@ $< $(CPPFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -f tutorial tutorial.o
