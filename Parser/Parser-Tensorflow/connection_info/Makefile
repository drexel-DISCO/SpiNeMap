all:
	g++ -std=c++17 -O3 -I/usr/include/hdf5/serial/ -Iinclude/ ncc.cc model.cc -o ncc -lhdf5_serial -lboost_system -lboost_filesystem
debug:
	g++ -g -std=c++17 -I/usr/include/hdf5/serial/ -Iinclude/ ncc.cc model.cc -o ncc -lhdf5_serial -lboost_system -lboost_filesystem -DDEBUG
clean:
	rm -rf ncc
