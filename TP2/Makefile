# ---------------------------- #
# -- SortieDeCache Makefile -- #
# ---------------------------- #

# -- File list ----------
FILE = openmp.cpp container.cpp ios.cpp pixmap_io.cpp csv.cpp


# -- Paths ----------
INC_PATH = include

# -- Macros ----------
CC = g++

# -- Flags ----------
# No Vectorize
C_OPTIMISATION_FLAGS = -O3 -w -std=c++17 -march=native -fopenmp -fno-tree-vectorize 

C_INC_FLAGS = -I$(INC_PATH)


CFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)
LDFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)

# --- Save Sobel filtered Images ----
CFLAGS+=-DSAVE_IMG

# ----- What exercices to bench -----
CFLAGS+=-DEX1
CFLAGS+=-DEX2
CFLAGS+=-DEX3
CFLAGS+=-DEX5

# ---- Save CSV benchmarks ------
CFLAGS+=-DSAVE_BENCH

# -- Final product ----------
PRODUCT = openmp.run

# -- src and obj List ----------
SRC = $(FILE)
OBJ = $(addsuffix .o, $(basename $(FILE)))

# -- Base rules ----------
%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

#-----Main rule ----------
$(PRODUCT): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(INC) -lm

# -- Cleanup ----------
clean:
	rm -f $(OBJ)
	rm -f $(PRODUCT)
	# rm -f *.csv