#pragma once

#include <vector>
#include "../FLAB/Tensor.h"

using namespace std;
using namespace FLAB;

namespace Coeus {

class NeuralGroup
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
    NeuralGroup(NeuralGroup& p_copy);
    ~NeuralGroup(void);

	void integrate(Tensor* p_input, Tensor* p_weights);
    void activate();

     string	getId() const { return _id; };
    int		getDim() const { return _dim; };

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
};

}
