#include "NeuralGroup.h"
#include "IDGen.h"
#include "LinearActivation.h"
#include "BinaryActivation.h"
#include "SigmoidActivation.h"
#include "TanhActivation.h"
#include "SoftplusActivation.h"
#include "ReluActivation.h"
#include "SoftmaxActivation.h"

using namespace std;
using namespace Coeus;
/**
 * NeuralGroup constructor creates layer of p_dim neurons with p_activationFunction
 * @param p_dim dimension of layer
 * @param p_activation_function get_type of activation function
 * @param p_bias
 */
NeuralGroup::NeuralGroup(const int p_dim, const ACTIVATION p_activation_function, const bool p_bias)
{
    _id = IDGen::instance().next();
	_bias_flag = p_bias;	
	
    _dim = p_dim;
    _activation_function = p_activation_function;
	init_activation_function();


	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
	_bias = Tensor::Random({ _dim}, 1);
}

NeuralGroup::NeuralGroup(nlohmann::json p_data) {
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_activation_function = static_cast<ACTIVATION>(p_data["actfn"].get<int>());
	init_activation_function();
	_bias_flag = p_data["bias"].get<bool>();
	//#TODO doriesit _bias = ;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
}

NeuralGroup::NeuralGroup(NeuralGroup &p_copy) {
    _id = p_copy._id;
    _dim = p_copy._dim;
    _activation_function = p_copy._activation_function;
	init_activation_function();
	_bias = Tensor(p_copy._bias);

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
    _bias_flag = p_copy._bias_flag;
}

/**
 * NeuralGroup destructor
 */
NeuralGroup::~NeuralGroup(void)
{
	if (_f != nullptr) delete _f;
}

/**
 * performs product of weights and input which is stored in actionPotential vector
 * @param p_input vector of input values
 * @param p_weights matrix of input connection params
 */
void NeuralGroup::integrate(Tensor* p_input, Tensor* p_weights) {
    _ap += (*p_weights) * (*p_input);
}

/**
 * calculates the get_output of layer according to activation function
 */
void NeuralGroup::activate() {
	if (is_bias()) {
		_ap += _bias;
	}

	_output = _f->activate(_ap);
	_ap.fill(0);
}

void NeuralGroup::set_output(Tensor* p_output) const {
    _output.override(p_output);
}

void NeuralGroup::update_bias(Tensor& p_delta_b) {
	_bias += p_delta_b;
}

void NeuralGroup::init_activation_function() {
	switch (_activation_function) {
	case LINEAR:
		_f = new LinearActivation();
		break;
	case BINARY:
		_f = new BinaryActivation();
		break;
	case SIGMOID:
		_f = new SigmoidActivation();
		break;
	case TANH:
		_f = new TanhActivation();
		break;
	case SOFTPLUS:
		_f = new SoftplusActivation();
		break;
	case RELU:
		_f = new ReluActivation();
		break;
	case SOFTMAX:
		_f = new SoftmaxActivation();
		break;
	default:
		_f = nullptr;
	}
}

