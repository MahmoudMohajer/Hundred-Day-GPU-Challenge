# Compiler
NVCC = nvcc

# Flags
FLAGS = -lineinfo

# Default source file (can be overridden)
SOURCE ?= program.cu

# Output executable (derived from source file)
PROGRAM = $(basename $(SOURCE)).out

# Default target
all: $(PROGRAM)

# Rule to build
$(PROGRAM): $(SOURCE)
	$(NVCC) $(FLAGS) $(SOURCE) -o $(PROGRAM)

# Clean up
clean:
	rm -f *.out