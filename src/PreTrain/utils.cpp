#include "utils.h"
#include "common.h"
#include <cassert>
#include <string>

Utils::Utils(int argc, char const *argv[]) {
	assert((argc & 1) && "argument count error");
	for (int i = 1; i < argc; i += 2) {
		arguments_.insert(pair<string, string>(argv[i], argv[i + 1]));
	}
	cerr << "Read arguments done" << TimeString() << endl;
	cout << "usage:bigclam.exe" << endl;
}

int Utils::ParseArgumentInt(string identifier, int init_argument, string usage){
	cout << '\t' << identifier << ':' << usage << "(default: " << init_argument << ")";
	map<string, string>::iterator it = arguments_.find(identifier);
	if(it == arguments_.end()){
		cout << "=\n";
		return init_argument;
	}
	cout << '=' << stoi(it->second) << '\n';
	return stoi(it->second);
}

string Utils::ParseArgumentString(string identifier, string init_argument, string usage){
	cout << '\t' << identifier << ':' << usage << "(default: " << init_argument << ")";
	map<string, string>::iterator it = arguments_.find(identifier);
	if(it == arguments_.end()){
		cout << "=\n";
		return init_argument;
	}
	cout << '=' << it->second << '\n';
	return it->second;	
}

double Utils::ParseArgumentDouble(string identifier, double init_argument, string usage){
	cout << '\t' << identifier << ':' << usage << "(default: " << init_argument << ")";
	map<string, string>::iterator it = arguments_.find(identifier);
	if(it == arguments_.end()){
		cout << "=\n";
		return init_argument;
	}
	cout << '=' << stod(it->second) << '\n';
	return stod(it->second);
}

// int main(int argc, char const *argv[])
// {
	
// 	return 0;
// }
