#pragma once
#include <initializer_list>

using namespace std;

namespace FLAB {

class __declspec(dllexport) Tensor
{
public:
	enum INIT {
		ZERO = 0,
		ONES = 1,
		VALUE = 2,
		RANDOM = 3
	};

	Tensor();
	Tensor(const initializer_list<int> p_shape, INIT p_init, double p_value = 1.);
	Tensor(const initializer_list<int> p_shape, double* p_data);
	Tensor(int p_rank, int* p_shape, double* p_data);
	Tensor(const initializer_list<int> p_shape, initializer_list <double> p_inputs);
	Tensor(const Tensor &p_copy);
	~Tensor();

	static Tensor Zero(const initializer_list<int> p_shape);
	static Tensor Ones(const initializer_list<int> p_shape);
	static Tensor Value(const initializer_list<int> p_shape, double p_value);
	static Tensor Random(const initializer_list<int> p_shape, double p_limit);
	static Tensor Concat(Tensor& p_vector1, Tensor& p_vector2);

	void operator = (const Tensor& p_tensor);
	Tensor operator + (const Tensor& p_tensor);
	void operator += (const Tensor& p_tensor);
	Tensor operator - (const Tensor& p_tensor);
	void operator -= (const Tensor& p_tensor);
	Tensor operator * (const Tensor& p_tensor);
	Tensor operator * (const double p_const);
	void operator *= (const double p_const);

	Tensor apply(double(*f)(double)) const;

	int max_index() const;

	void fill(double p_value) const;

	double sum();

	int size() const { return _size; }

	double at(int p_x) const;
	double at(int p_y, int p_x) const;
	void set(int p_x, double p_val) const;
	void set(int p_y, int p_x, double p_val) const;

private:
	void init_shape(int p_rank, int* p_shape);
	void init_shape(initializer_list<int> p_shape);
	void fill(INIT p_init, double p_value) const;
	
	double *_arr;
	int _rank;
	int *_shape;
	int _size;
};

}
