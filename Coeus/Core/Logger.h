#pragma once

#include <string>
#include <fstream>

using namespace std;

namespace Coeus {

class __declspec(dllexport) LoggerInstance
{
	public:
		LoggerInstance(const string& p_name = "");
		LoggerInstance(LoggerInstance& p_copy);
		LoggerInstance& operator=(const LoggerInstance& p_copy);
		~LoggerInstance() = default;

		void init(const string& p_dir = "");
		void log(const string& p_msg);
		void close();

	private:
		string _directory;
		string _filename;
		ofstream _file;
};

class __declspec(dllexport) Logger
{
public:
	static Logger& instance();
	LoggerInstance init(const string& p_name = "") const;

private:
	Logger();
	~Logger();
};

}

