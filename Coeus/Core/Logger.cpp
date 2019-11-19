#include "Logger.h"

using namespace Coeus;



Logger::Logger()
{
}


Logger::~Logger()
{
}

Logger& Logger::instance() 
{
	static Logger logger;
	return logger;
}

void Logger::init(string p_name)
{
	_file.open(p_name);
}

void Logger::log(string p_msg) 
{
	_file << p_msg << endl;
}

void Logger::close() 
{
	_file.close();
}
