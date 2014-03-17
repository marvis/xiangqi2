all:
	g++ -o chessboardseg chessboardseg.cpp `pkg-config --cflags --libs opencv`
	#g++ -o edgemap edgemap.cpp `pkg-config --cflags --libs opencv`
	#g++ -o findlines findlines.cpp `pkg-config --cflags --libs opencv`
	#g++ -o findcircles findcircles.cpp `pkg-config --cflags --libs opencv`
	#g++ -o squaresum squaresum.cpp `pkg-config --cflags --libs opencv`
