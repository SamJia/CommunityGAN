#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <map>
#include <iostream>
using namespace std;



class Utils
{
public:
	Utils(int argc, char const *argv[]);
	~Utils() = default;
	int ParseArgumentInt(string arg, int init_argument, string usage);
	string ParseArgumentString(string arg, string init_argument, string usage);
	double ParseArgumentDouble(string arg, double init_argument, string usage);
private:
	map<string, string> arguments_;	
};
#endif
