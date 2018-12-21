# compiler
CC=g++ 
CFLAGS=-c -Wall

all: hello

hello: main.o Graph.o Model.o Unigram.o
	$(CC) main.o Graph.o Model.o Unigram.o -o hello

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

Graph.o: Graph.cpp
	$(CC) $(CFLAGS) Graph.cpp

hello.o: Model.cpp
	$(CC) $(CFLAGS) Model.cpp

hello.o: Unigram.cpp
	$(CC) $(CFLAGS) Unigram.cpp

clean:
	rm *o hello
