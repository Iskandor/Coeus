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
    _dim = p_dim;
	if (_bias) _dim += 1;
    _activationFunction = p_activationFunction;

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
}

NeuralGroup::NeuralGroup(nlohmann::json p_data) {
	_id = p_data["id"].get<string>();
	_dim = p_data["dim"].get<int>();
	_activationFunction = static_cast<ACTIVATION>(p_data["actfn"].get<int>());

	_output = Tensor::Zero({ _dim });
	_ap = Tensor::Zero({ _dim });
}

NeuralGroup::NeuralGroup(NeuralGroup &p_copy) {
    _id = p_copy._id;
    _dim = p_copy._dim;
    _activationFunction = p_copy._activationFunction;

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
 * calculates the output of layer according to activation function
 */
void NeuralGroup::activate() {
    switch (_activationFunction) {
        case IDENTITY:
        case LINEAR:
			_output = _ap.apply(ActivationFunctions::linear);
            break;
        case BINARY:
			_output = _ap.apply(ActivationFunctions::binary);
            break;
        case SIGMOID:
			_output = _ap.apply(ActivationFunctions::sigmoid);
            break;
        case TANH:
			_output = _ap.apply(ActivationFunctions::tanh);
            break;
        case SOFTPLUS:
			_output = _ap.apply(ActivationFunctions::softplus);
            break;
        case RELU:
			_output = _ap.apply(ActivationFunctions::relu);
            break;
	    default: ;
    }
	_ap.fill(0);
}

void NeuralGroup::setOutput(Tensor* p_output) {
    _output.override(p_output);
}

