#include "NadamRule.h"

using namespace Coeus;


NadamRule::NadamRule(NetworkGradient* p_network_gradient, const double p_alpha, const double p_beta1, const double p_beta2,
                     const double p_epsilon): IUpdateRule(p_network_gradient, p_alpha), _beta1(p_beta1), _beta2(p_beta2), _epsilon(p_epsilon)
{
	_m = p_network_gradient->get_empty_params();
	_m_mean = p_network_gradient->get_empty_params();
	_v = p_network_gradient->get_empty_params();
	_v_mean = p_network_gradient->get_empty_params();
}

NadamRule::~NadamRule()
= default;

IUpdateRule* NadamRule::clone(NetworkGradient* p_network_gradient)
{
	return new NadamRule(p_network_gradient, _alpha, _beta1, _beta2, _epsilon);
}

void NadamRule::merge(IUpdateRule** p_rule, const int p_size)
{

	for(int i = 0; i < p_size; i++)
	{
		auto rule = dynamic_cast<NadamRule*>(p_rule[i]);

		map<string, Tensor>* g = rule->_network_gradient->get_gradient();

		for (auto key = _update.begin(); key != _update.end(); ++key)
		{
			Tensor* m = &_m[key->first];
			Tensor* v = &_v[key->first];
			
			for (int i = 0; i < g->at(key->first).size(); i++) {
				(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * g->at(key->first)[i];
				(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow(g->at(key->first)[i], 2);
			}
		}
	}

	for(auto key = _update.begin(); key != _update.end(); ++key)
	{		
		Tensor* m = &_m[key->first];
		Tensor* v = &_v[key->first];
		Tensor* m_mean = &_m_mean[key->first];
		Tensor* v_mean = &_v_mean[key->first];

		for (int i = 0; i < _update[key->first].size(); i++) {
			(*m_mean)[i] = (*m)[i] / (1 - _beta1);
			(*v_mean)[i] = (*v)[i] / (1 - _beta2);
		}
	}
}

void NadamRule::reset()
{
	for (auto key = _update.begin(); key != _update.end(); ++key)
	{
		_m[key->first].fill(0);
		_v[key->first].fill(0);
		_m_mean[key->first].fill(0);
		_v_mean[key->first].fill(0);
	}
}

void NadamRule::override(IUpdateRule* p_rule)
{
	NadamRule* rule = static_cast<NadamRule*>(p_rule);
	for (auto key = _update.begin(); key != _update.end(); ++key)
	{
		_m[key->first] = rule->_m[key->first];
		_v[key->first] = rule->_v[key->first];
		_m_mean[key->first] = rule->_m_mean[key->first];
		_v_mean[key->first] = rule->_v_mean[key->first];
	}
}

void NadamRule::update_momentum(const string& p_id, Tensor & p_gradient)
{
	Tensor* m = &_m[p_id];
	Tensor* v = &_v[p_id];
	Tensor* m_mean = &_m_mean[p_id];
	Tensor* v_mean = &_v_mean[p_id];

	for (int i = 0; i < p_gradient.size(); i++) {
		(*m)[i] = _beta1 * (*m)[i] + (1 - _beta1) * p_gradient[i];
		(*v)[i] = _beta2 * (*v)[i] + (1 - _beta2) * pow(p_gradient[i], 2);
		(*m_mean)[i] = (*m)[i] / (1 - _beta1);
		(*v_mean)[i] = (*v)[i] / (1 - _beta2);
	}
}

void NadamRule::calc_update(map<string, Tensor>* p_gradient)
{
	IUpdateRule::calc_update(p_gradient);

	for (auto it = p_gradient->begin(); it != p_gradient->end(); ++it) {

		update_momentum(it->first, it->second);

		Tensor* update = &_update[it->first];
		Tensor* m_mean = &_m_mean[it->first];
		Tensor* v_mean = &_v_mean[it->first];

		for (int i = 0; i < it->second.size(); i++) {
			(*update)[i] = -_alpha / (sqrt((*v_mean)[i]) + _epsilon) * (_beta1 * (*m_mean)[i] + (1 - _beta1) * it->second[i] / (1 - _beta1));
		}
	}
}