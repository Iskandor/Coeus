#pragma once
#include "NeuralGroup.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) Connection
{
public:
    enum INIT {
        UNIFORM = 0,
        LECUN_UNIFORM = 1,
        GLOROT_UNIFORM = 2,
        IDENTITY = 3
    };

	Connection(int p_inDim, int p_outDim, string p_inId, string p_outId);
	explicit Connection(nlohmann::json p_data);
    Connection(Connection& p_copy);
    ~Connection(void);

    void init(INIT p_init, double p_limit);
    void init(Tensor* p_weights);
    void set_weights(Tensor* p_weights) const;
    Tensor* get_weights() const { return _weights; };
	void update_weights(Tensor& p_delta_w) const;

	string get_id() const { return _id; };
	string get_in_id() const { return _in_id; };
	string get_out_id() const { return _out_id; };
	int get_in_dim() const { return _in_dim; };
	int get_out_dim() const { return _out_dim; };

private:
	
    void uniform(double p_limit);
    void identity();


    string _id;
	string _in_id, _out_id;
    int _in_dim, _out_dim;
    Tensor *_weights;
};

}