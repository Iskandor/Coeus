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
	LoggerInstance instance(p_name);
	return instance;
}

LoggerInstance::LoggerInstance(const string & p_name)
{
	string name = p_name;

	if (p_name.empty())
	{
		const time_t timestamp = std::time(nullptr);
		name = "log" + to_string(timestamp) + ".log";
	}

	_filename = name;
	_directory = "";
}

LoggerInstance::LoggerInstance(LoggerInstance& p_copy)
{
	_filename = p_copy._filename;
	_directory = p_copy._directory;
	_file.open(p_copy._filename);
}

LoggerInstance& LoggerInstance::operator=(const LoggerInstance& p_copy)
{
	_filename = p_copy._filename;
	_directory = p_copy._directory;
	_file.open(p_copy._filename);
	return *this;
}

void LoggerInstance::init(const string& p_dir)
{
	_directory = p_dir;
	_file.open(_directory + _filename);
}

void LoggerInstance::log(const string& p_msg)
{
	_file << p_msg << endl;
}

void LoggerInstance::close()
{
	_file.close();
}
