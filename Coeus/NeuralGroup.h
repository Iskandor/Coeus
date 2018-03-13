#pragma once

#include <Tensor.h>
#include "json.hpp"

using namespace std;
using namespace FLAB;

namespace Coeus {

class __declspec(dllexport) NeuralGroup
{
public:
    enum ACTIVATION {
     IDENTITY = 0,
     BIAS = 1,
     BINARY = 2,
     SIGMOID = 3,
     TANH = 4,
     LINEAR = 6,
     EXPONENTIAL = 7,
     SOFTPLUS = 8,
     RELU = 9,
     KEXPONENTIAL = 10,
     GAUSS = 11
    };

    NeuralGroup(int p_dim, ACTIVATION p_activationFunction, bool p_bias);
	explicit NeuralGroup(nlohmann::json p_data);
    NeuralGroup(NeuralGroup& p_copy);
    ~NeuralGroup(void);

	void integrate(Tensor* p_input, Tensor* p_weights);
    void activate();

     string	getId() const { return _id; };
    int		getDim() const { return _dim; };
	bool	is_bias() const { return _bias; };

    void	setOutput(Tensor* p_output);
    Tensor* getOutput() { return &_output; };

    ACTIVATION getActivationFunction() const { return _activationFunction; };

private:
    string  _id;
    ACTIVATION _activationFunction;

	int     _dim;
    Tensor	_output;
    Tensor	_ap;
	bool	_bias;
	int		_bias_index;
};

}
