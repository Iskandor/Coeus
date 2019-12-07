#pragma once
#include <cstdint>
#include <string>

namespace Coeus
{
	
class __declspec(dllexport) IDGen
{
public:
	static IDGen& instance();
	std::string next();
private:
	IDGen() : _id(0) {}
	uint32_t _id;
};

}

