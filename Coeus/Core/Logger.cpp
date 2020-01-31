#include <ctime>
#include "Logger.h"

using namespace Coeus;



Logger::Logger()
= default;


Logger::~Logger()
= default;

Logger& Logger::instance() 
{
	static Logger logger;
	return logger;
}

void Logger::init(const string& p_name)
{
	if (p_name.empty())
	{
		const int timestamp = std::time(nullptr);
		_file.open("log" + to_string(timestamp) + ".log");
	}
	else
	{
		_file.open(p_name);
	}
	
}

void Logger::log(const string& p_msg) 
{
	_file << p_msg << endl;
}

void Logger::close() 
{
	_file.close();
}
