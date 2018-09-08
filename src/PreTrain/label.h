#ifndef LABEL_H
#define LABEL_H
#include <string>
using namespace std;

struct Label {
	int date;
	string name;
	Label(string n = "", int d = 0) : name(n), date(d) {}
	bool operator<(const Label &label2) const {
		return date < label2.date;
	}
};

#endif