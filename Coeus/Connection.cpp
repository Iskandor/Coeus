#include <random>
#include "Connection.h"

using namespace Coeus;

Connection::Connection(const int p_in_dim, const int p_out_dim, const string p_in_id, const string p_out_id)
{
    _id = p_in_id + "_" + p_out_id;
	_in_dim = p_in_dim;
    _out_dim = p_out_dim;
	_in_id = p_in_id;
	_out_id = p_out_id;
	_weights = nullptr;
}

Connection::Connection(Connection &p_copy) {
    _id = p_copy._id;
    _in_dim = p_copy._in_dim;
    _out_dim = p_copy._out_dim;
    _weights = new Tensor(*p_copy._weights);
}

Connection::~Connection()
{
	if (_weights != nullptr) delete _weights;
}

void Connection::init(const Connection::INIT p_init, const double p_limit) {
    switch(p_init) {
        case UNIFORM:
            uniform(p_limit);
            break;
        case LECUN_UNIFORM:
            uniform(static_cast<double>(pow(_in_dim, -.5)));
            break;
        case GLOROT_UNIFORM:
            uniform(2 / (_in_dim + _out_dim));
            break;
        case IDENTITY:
            identity();
            break;
    }
}

void Connection::init(Tensor *p_weights) {
    _weights = p_weights;
}

void Connection::uniform(const double p_limit) {
	_weights = new Tensor({ _out_dim, _in_dim }, Tensor::RANDOM, p_limit);
}

void Connection::identity() {
	_weights = new Tensor({ _out_dim, _in_dim }, Tensor::ONES);
}

void Connection::set_weights(Tensor *p_weights) {
    delete _weights;
    _weights = new Tensor(*p_weights);
}

void Connection::update_weights(Tensor& p_delta_w) {
	*_weights += p_delta_w;
}
