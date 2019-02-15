#include "ReplayBuffer.h"
#include "RandomGenerator.h"

using namespace Coeus;

ReplayBuffer::ReplayBuffer(const int p_size) {
	_size = p_size;
}

ReplayBuffer::~ReplayBuffer()
{
	for (int i = 0; i < _buffer.size(); i++) {
		delete _buffer[i];
	}
}

void ReplayBuffer::add_item(Tensor* p_s0, const float p_a, Tensor* p_s1, const float p_r, const bool p_final) {
	if (_buffer.size() > _size) {
		delete _buffer[0];
		_buffer.erase(_buffer.begin());
	}
	_buffer.push_back(new Item(p_s0, p_a, p_s1, p_r, p_final));
}

vector<ReplayBuffer::Item*>* ReplayBuffer::get_sample(const int p_size) {
	int size = p_size;
	_sample.clear();

	if (size > _buffer.size()) size = _buffer.size();

	vector<int> index = RandomGenerator::getInstance().choice(_buffer.size(), size);

	for(int i = 0; i < index.size(); i++) {
		_sample.push_back(_buffer[i]);
	}

	return &_sample;
}
