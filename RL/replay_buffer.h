#pragma once
#include <random>
#include <vector>
#include <algorithm>
#include "random_generator.h"

struct COEUS_DLL_API mdp_transition
{
	tensor s0;
	tensor a;
	tensor s1;
	float r;
	bool final;

	mdp_transition(tensor& p_s0, tensor& p_a, tensor& p_s1, const float p_r, const bool p_final) {
		s0 = p_s0;
		a = p_a;
		s1 = p_s1;
		r = p_r;
		final = p_final;
	}
};

template <typename T>
class COEUS_DLL_API replay_buffer
{
public:
	replay_buffer(const int p_size) {
		_size = p_size;
	}

	~replay_buffer() = default;

	void add_item(T p_item) {
		if (_buffer.size() == _size) {
			_buffer.erase(_buffer.begin());
		}
		_buffer.push_back(p_item);
	}

	std::vector<T*>& sample(const size_t p_size) {
		size_t size = p_size;
		_sample.clear();

		if (size > _buffer.size()) size = _buffer.size();

		std::vector<int> index = random_generator::instance().choice(_buffer.size(), size);
		std::shuffle(index.begin(), index.end(), std::mt19937(std::random_device()()));

		for (int i = 0; i < index.size(); i++) {
			_sample.push_back(&_buffer[index[i]]);
		}

		return _sample;
	}

	size_t size() const
	{
		return _buffer.size();
	}

private:
	std::vector<T>	_buffer;
	std::vector<T*>	_sample;
	size_t _size;


};

