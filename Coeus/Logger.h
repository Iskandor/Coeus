#pragma once

#include <string>
#include <fstream>

using namespace std;

namespace Coeus {

class __declspec(dllexport) Logger
{
public:
	static Logger& instance();
	void init(string p_name);
	void log(string p_msg);
	void close();

private:
	Logger();
	~Logger();

	ofstream _file;
};

}

