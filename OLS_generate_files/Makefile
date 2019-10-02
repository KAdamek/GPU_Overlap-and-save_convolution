GCC = g++

GCC_OPTS =-O3 -Wall -Wextra $(INC)

ANALYZE = Example_files.exe

all: clean analyze

analyze: Makefile
	$(GCC) -o $(ANALYZE) Example_files.cpp $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


