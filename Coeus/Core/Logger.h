#pragma once

#include <string>
#include <fstream>

using namespace std;

namespace Coeus {

class __declspec(dllexport) Logger
{
public:
	static Logger& instance();
	void init(const string& p_name = "");
	void log(const string& p_msg);
	void close();

private:
	Logger();
	~Logger();

	ofstream _file;
};

}

