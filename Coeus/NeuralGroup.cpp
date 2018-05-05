#include "NeuralGroup.h"
#include "ActivationFunctions.h"
#include "IDGen.h"

using namespace std;
using namespace Coeus;
/**
 * NeuralGroup constructor creates layer of p_dim neurons with p_activationFunction
 * @param p_id name of layer must be unique per network
 * @param p_dim dimension of layer
 * @param p_activationFunction type of activation function
 */
NeuralGroup::NeuralGroup(int p_dim, ACTIVATION p_activationFunction, bool p_bias)
{
    _id = IDGen::instance().next();
	_bias = p_bias;	
	_bias_index = -1;
    _dim = p_dim;
	if (_bias) {
		_dim += 1;
		_bias_index = _dim - 1;
	}
    _activationFunction = p_activationFunction;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
}

NeuralGroup::NeuralGroup(nlohmann::json p_data) {
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_activationFunction = static_cast<ACTIVATION>(p_data["actfn"].get<int>());
	_bias = p_data["bias"].get<bool>();
	_bias_index = _dim - 1;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
}

NeuralGroup::NeuralGroup(NeuralGroup &p_copy) {
    _id = p_copy._id;
    _dim = p_copy._dim;
    _activationFunction = p_copy._activationFunction;
	_bias_index = p_copy._bias_index;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
    _bias = p_copy._bias;
}

/**
 * NeuralGroup destructor frees filters
 */
NeuralGroup::~NeuralGroup(void)
{

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
    switch (_activationFunction) {
        case IDENTITY:
        case LINEAR:
			_output = Tensor::apply(_ap, ActivationFunctions::linear);
            break;
        case BINARY:
			_output = Tensor::apply(_ap, ActivationFunctions::binary);
            break;
        case SIGMOID:
			_output = Tensor::apply(_ap, ActivationFunctions::sigmoid);
            break;
        case TANH:
			_output = Tensor::apply(_ap, ActivationFunctions::tanh);
            break;
        case SOFTPLUS:
			_output = Tensor::apply(_ap, ActivationFunctions::softplus);
            break;
        case RELU:
			_output = Tensor::apply(_ap, ActivationFunctions::relu);
            break;
	    default: ;
    }
	_ap.fill(0);

	if (_bias) {
		_output.set(_bias_index, 1.0);
	}
}

void NeuralGroup::set_output(Tensor* p_output) const {
    _output.override(p_output);
}

