cc = g++
exe = magic
std = c++11
objects = ./model.o ./magic.o ./nnvector.o ./common.o ./utils.o ./graph.o

$(exe) : $(objects)
			$(cc) -O2 -fopenmp -std=$(std) -o $(exe) $(objects)

./model.o : ./model.cpp model.h nnvector.h graph.h label.h utils.h common.h
	$(cc) -std=$(std) -O2 -fopenmp -c ./model.cpp -o ./model.o
./magic.o : ./magic.cpp model.h nnvector.h graph.h label.h utils.h
	$(cc) -std=$(std) -O2 -fopenmp -c ./magic.cpp -o ./magic.o
./nnvector.o : ./nnvector.cpp nnvector.h
	$(cc) -std=$(std) -O2 -fopenmp -c ./nnvector.cpp -o ./nnvector.o
./common.o : ./common.cpp common.h nnvector.h
	$(cc) -std=$(std) -O2 -fopenmp -c ./common.cpp -o ./common.o
./utils.o : ./utils.cpp common.h nnvector.h utils.h
	$(cc) -std=$(std) -O2 -fopenmp -c ./utils.cpp -o ./utils.o
./graph.o : ./graph.cpp label.h graph.h
	$(cc) -std=$(std) -O2 -fopenmp -c ./graph.cpp -o ./graph.o

clean :
		rm $(objects) $(exe)
