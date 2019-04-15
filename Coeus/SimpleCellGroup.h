#pragma once

#include "Tensor.h"
#include "json.hpp"
#include "Coeus.h"
#include "BaseCellGroup.h"

using namespace std;
using namespace FLAB;

namespace Coeus {

class __declspec(dllexport) SimpleCellGroup : public BaseCellGroup
{
public:
    SimpleCellGroup(int p_dim, ACTIVATION p_activation_function, bool p_bias);
	explicit SimpleCellGroup(json p_data);
    SimpleCellGroup(SimpleCellGroup& p_copy);
	explicit SimpleCellGroup(SimpleCellGroup* p_source);
	SimpleCellGroup& operator = (const SimpleCellGroup& p_copy);
    ~SimpleCellGroup();

	void integrate(Tensor* p_input, Tensor* p_weights) override;
    void activate() override;

	json get_json() const override;
};
}
