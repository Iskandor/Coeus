#include "SimpleCellGroup.h"

using namespace std;
using namespace Coeus;
/**
 * SimpleCellGroup constructor creates layer of p_dim neurons with p_activationFunction
 * @param p_dim dimension of layer
 * @param p_activation_function get_type of activation function
 * @param p_bias
 */
SimpleCellGroup::SimpleCellGroup(const int p_dim, const ACTIVATION p_activation_function, const bool p_bias) : BaseCellGroup(p_dim, p_bias)
{
	_f = init_activation_function(p_activation_function);
}

SimpleCellGroup::SimpleCellGroup(nlohmann::json p_data): BaseCellGroup(p_data)
{
	_f = init_activation_function(static_cast<ACTIVATION>(p_data["actfn"].get<int>()));
}

SimpleCellGroup::SimpleCellGroup(SimpleCellGroup &p_copy) : BaseCellGroup(p_copy._dim, p_copy._bias_flag)
{
	_f = init_activation_function(p_copy._f->get_type());
	_bias = Tensor(p_copy._bias);
	_bias_flag = p_copy._bias_flag;
}

SimpleCellGroup& SimpleCellGroup::operator=(const SimpleCellGroup& p_copy)
{
	copy(p_copy);
	_f = init_activation_function(p_copy._f->get_type());
	_bias = Tensor(p_copy._bias);
	_bias_flag = p_copy._bias_flag;

	return *this;
}

/**
 * NeuralGroup destructor
 */
SimpleCellGroup::~SimpleCellGroup()
{
	delete _f;
}

/**
 * performs product of weights and input which is stored in actionPotential vector
 * @param p_input vector of input values
 * @param p_weights matrix of input connection params
 */
void SimpleCellGroup::integrate(Tensor* p_input, Tensor* p_weights) {
    _net += (*p_weights) * (*p_input);
}

/**
 * calculates the get_output of layer according to activation function
 */
void SimpleCellGroup::activate() {
	if (is_bias()) {
		_net += _bias;
	}

	_output = _f->activate(_net);
	_deriv_output = _f->deriv(_output);
	_net.fill(0);
}

SimpleCellGroup* SimpleCellGroup::clone()
{
	return new SimpleCellGroup(*this);
}
