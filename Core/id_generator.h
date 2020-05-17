#pragma once
#include <cstdint>
#include <string>
#include "base64.h"

class id_generator
{
public:
	static std::string next()
	{
		std::string res;
		Base64::Encode(std::to_string(_id), &res);
		_id++;

		return res;
	}
private:
	id_generator() = default;
	~id_generator() = default;

	static uint32_t _id;
};

uint32_t id_generator::_id = 0;