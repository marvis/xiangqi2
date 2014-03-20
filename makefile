all:
	g++ -o fitlines fitlines.cpp `pkg-config --cflags --libs opencv`
	#g++ -o houghlines houghlines.cpp `pkg-config --cflags --libs opencv`
	#g++ -o kmeanslines kmeanslines.cpp `pkg-config --cflags --libs opencv`
	#g++ -o chessgrid chessgrid.cpp `pkg-config --cflags --libs opencv`
	#g++ -o findcorners findcorners.cpp `pkg-config --cflags --libs opencv`
	#g++ -o fillbkg fillbkg.cpp `pkg-config --cflags --libs opencv`
	#g++ -o chessboardseg chessboardseg.cpp `pkg-config --cflags --libs opencv`
	#g++ -o edgemap edgemap.cpp `pkg-config --cflags --libs opencv`
	#g++ -o findlines findlines.cpp `pkg-config --cflags --libs opencv`
	#g++ -o findcircles findcircles.cpp `pkg-config --cflags --libs opencv`
	#g++ -o squaresum squaresum.cpp `pkg-config --cflags --libs opencv`
