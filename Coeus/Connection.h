#pragma once
#include "SimpleCellGroup.h"

using namespace std;

namespace Coeus {

class __declspec(dllexport) Connection : public ParamModel
{
public:
    enum INIT {
		NONE = 0,
        UNIFORM = 1,
        LECUN_UNIFORM = 2,
        GLOROT_UNIFORM = 3,
        IDENTITY = 4
    };

	Connection(int p_in_dim, int p_out_dim, const string& p_in_id, const string& p_out_id, bool p_trainable = true);
	Connection(int p_in_dim, int p_out_dim, const string& p_in_id, const string& p_out_id, INIT p_init, bool p_trainable = true, double p_limit = 0);
	enum NORM
	{
		L1_NORM = 1,
		L2_NORM = 2
	};

	explicit Connection(nlohmann::json p_data);
	Connection* clone() const;
    ~Connection(void);

    void set_weights(Tensor* p_weights) const;
    Tensor* get_weights() const { return _weights; };
	void update_weights(Tensor& p_delta_w) const;
	void normalize_weights(NORM p_norm) const;
	void override(Connection* p_copy);

	string get_id() const { return _id; };
	string get_in_id() const { return _in_id; };
	string get_out_id() const { return _out_id; };
	int get_in_dim() const { return _in_dim; };
	int get_out_dim() const { return _out_dim; };
	bool is_trainable() const { return _trainable; };

	json get_json() const;

private:
	void init(INIT p_init, bool p_trainable, double p_limit);
    void uniform(double p_limit);
    void identity();

    string _id;
	string _in_id, _out_id;
    int _in_dim, _out_dim;
    Tensor* _weights;
	Tensor _norm;
	bool _trainable;
};

}