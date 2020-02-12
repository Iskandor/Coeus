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

LoggerInstance Logger::init(const string& p_name) const
{
	string name = p_name;
	
	if (p_name.empty())
	{
		const time_t timestamp = std::time(nullptr);
		name = "log" + to_string(timestamp) + ".log";
	}

	LoggerInstance instance(name);
	return instance;
}

LoggerInstance::LoggerInstance()
= default;

LoggerInstance::LoggerInstance(const string & p_name)
{
	_filename = p_name;
	_file.open(p_name);
}

LoggerInstance::LoggerInstance(LoggerInstance& p_copy)
{
	_filename = p_copy._filename;
	_file.open(p_copy._filename);
}

LoggerInstance& LoggerInstance::operator=(const LoggerInstance& p_copy)
{
	_filename = p_copy._filename;
	_file.open(p_copy._filename);
	return *this;
}

void LoggerInstance::log(const string& p_msg)
{
	_file << p_msg << endl;
}

void LoggerInstance::close()
{
	_file.close();
}
