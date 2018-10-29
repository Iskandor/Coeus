#include "RMSPropRule.h"

using namespace Coeus;

RMSPropRule::RMSPropRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_decay, const double p_epsilon):
	IUpdateRule(p_network_gradient, p_alpha), _decay(p_decay), _epsilon(p_epsilon)
{
	_cache = p_network_gradient->get_empty_params();
}

RMSPropRule::~RMSPropRule()
= default;

void RMSPropRule::calc_update() {
	IUpdateRule::calc_update();

	for (auto it = _network_gradient->get_w_gradient()->begin(); it != _network_gradient->get_w_gradient()->end(); ++it) {

		/*
		for(int i = 0; i < it->second.size(); i++) {
		_cache[it->first][i] = _decay * _cache[it->first][i] + (1 - _decay) * pow(it->second[i], 2);
		_update[it->first][i] = -_alpha / sqrt(_cache[it->first][i] + _epsilon) * it->second[i];
		}
		*/

		_cache[it->first] = _decay * _cache[it->first] + (1 - _decay) * it->second.pow(2);
		_update[it->first] = (-_alpha / (_cache[it->first] + _epsilon).sqrt()).dot(it->second);
	}
}

void RMSPropRule::merge(IUpdateRule** p_rule, int p_size)
{
	for (auto it = _cache.begin(); it != _cache.end(); ++it)
	{
		_cache[it->first].fill(0);
	}

	RMSPropRule** rule = reinterpret_cast<RMSPropRule**>(p_rule);

	for(int i = 0; i < p_size; i++)
	{
		for (auto it = _cache.begin(); it != _cache.end(); ++it)
		{
			_cache[it->first] += rule[i]->_cache[it->first];
		}
	}

	for (auto it = _cache.begin(); it != _cache.end(); ++it)
	{
		_cache[it->first] /= p_size;
	}
}

IUpdateRule* RMSPropRule::clone(NetworkGradient* p_network_gradient)
{
	RMSPropRule* result = new RMSPropRule(p_network_gradient, _alpha, _decay, _epsilon);

	result->_cache = p_network_gradient->get_empty_params();

	return result;
}