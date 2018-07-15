#include "Connection.h"

using namespace Coeus;

Connection::Connection(const int p_in_dim, const int p_out_dim, const string& p_in_id, const string& p_out_id)
{
    _id = p_out_id + "_" + p_in_id;
	_in_dim = p_in_dim;
    _out_dim = p_out_dim;
	_in_id = p_in_id;
	_out_id = p_out_id;
	_trainable = true;

	_norm = Tensor::Zero({ _out_dim });
}

Connection::Connection(nlohmann::json p_data) {
	_id = p_data["id"].get<string>();
	_in_id = p_data["in_id"].get<string>();
	_out_id = p_data["out_id"].get<string>();
	_in_dim = p_data["in_dim"].get<int>();
	_out_dim = p_data["out_dim"].get<int>();
	_trainable = p_data["trainable"].get<bool>();

	double* data = Tensor::alloc_arr(_out_dim * _in_dim);

	stringstream ss(p_data["weights"].get<string>());

	ss.seekg(0,ios::end);
	const streampos size = ss.tellg();
	ss.seekg(0, ios::beg);
	ss.read(reinterpret_cast<char*>(data), size);

	_weights = Tensor({_out_dim, _in_dim}, data);
}

Connection::Connection(Connection &p_copy) {
    _id = p_copy._id;
    _in_dim = p_copy._in_dim;
    _out_dim = p_copy._out_dim;
	_in_id = p_copy._in_id;
	_out_id = p_copy._out_id;
    _weights = Tensor(p_copy._weights);
}

Connection::~Connection()
{
}

void Connection::init(const INIT p_init, const double p_limit) {
    switch(p_init) {
		case NONE:
			_trainable = false;
			_weights = Tensor::Zero({ _out_dim, _in_dim });
			break;
        case UNIFORM:
            uniform(p_limit);
            break;
        case LECUN_UNIFORM:
            uniform(static_cast<double>(pow(_in_dim, -.5)));
            break;
        case GLOROT_UNIFORM:
            uniform(2.0f / (_in_dim + _out_dim));
            break;
        case IDENTITY:
			_trainable = false;
            identity();
            break;
    }
}

void Connection::uniform(const double p_limit) {
	_weights = Tensor({ _out_dim, _in_dim }, Tensor::RANDOM, p_limit);
}

void Connection::identity() {
	_weights = Tensor::Ones({ _out_dim, _in_dim });
}

void Connection::set_weights(Tensor *p_weights) const {
    _weights.override(p_weights);
}

void Connection::update_weights(Tensor& p_delta_w) {
	_weights += p_delta_w;
}

void Connection::normalize_weights(const NORM p_norm) const {

	switch (p_norm) {
	case L1_NORM:
		
		for (int i = 0; i < _out_dim; i++) {
			_norm[i] = 0;
			for (int j = 0; j < _in_dim; j++) {
				_norm[i] += abs(_weights.at(i, j));
			}
		}

		break;
	case L2_NORM:
		for (int i = 0; i < _out_dim; i++) {
			_norm[i] = 0;
			for (int j = 0; j < _in_dim; j++) {
				_norm[i] += pow(_weights.at(i, j), 2);
			}
			_norm[i] = sqrt(_norm[i]);
		}

		break;

	}

	for (int i = 0; i < _out_dim; i++) {
		for (int j = 0; j < _in_dim; j++) {
			if (_norm[i] > 0) {
				_weights.set(i, j, _weights.at(i, j) / _norm[i]);
			}			
		}
	}
}
