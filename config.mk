# select compiler (comment all for auto)
#CC = gcc
#CC = icc
#CC = pgcc
#CC = craycc
#CC = clang

# select MPI compiler (comment all for auto)
#MPICC = gcc
#MPICC = icc
#MPICC = pgcc
#MPICC = craycc
#MPICC = clang
#MPICC = mpicc

# select python (comment all for auto)
#PYTHON = python
#PYTHON = python3

# enable vectorizations (comment all for auto)
#MMX := true
#MMX := false
#SSE := true
#SSE := false
#SSE2 := true
#SSE2 := false
#SSE3 := true
#SSE3 := false
#SSE4_1 := true
#SSE4_1 := false
#SSE4_2 := true
#SSE4_2 := false
#AVX := true
#AVX := false
#AVX2 := true
#AVX2 := false

# optimization flags (comment for auto)
#OPTFLAGS := -O3

# Again, by default we are running on the same architecture we are
# building - if you're cross compiling then you should set this manually
ENDIAN := $(shell perl -le 'print unpack(N,pack(L,0x01020304)) == 0x01020304 ? big : little')

# select comparison BLAS (comment all for auto)
#BLAS = -lblas
#BLAS = -latlas
#BLAS = -lmkl_sequential
#BLAS = -framework Acclerate

# set build mode (comment all for auto)
#BUILD_MODE := release
#BUILD_MODE := debug
#BUILD_MODE := profile

ARGS = $(TOP)/src/default_args.json
PARAMS = $(TOP)/src/params.json

# Make the compiler invocation lines verbose - if it is not defined or
# set to value other then "true" you'll see just indication of what is
# being compiled (without details about options)
#VERBOSE := true

# Uncomment if you don't like coloring of the output
#COLOR_TTY := false
