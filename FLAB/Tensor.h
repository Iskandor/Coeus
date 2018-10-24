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
	Tensor(int p_rank, int* p_shape, INIT p_init, double p_value = 1.);
	Tensor(const initializer_list<int> p_shape, initializer_list <double> p_inputs);
	Tensor(const Tensor &p_copy);
	~Tensor();
	Tensor operator-() const;

	static Tensor Zero(const initializer_list<int> p_shape);
	static Tensor Ones(const initializer_list<int> p_shape);
	static Tensor Value(const initializer_list<int> p_shape, double p_value);
	static Tensor Random(const initializer_list<int> p_shape, double p_limit);
	static Tensor Concat(Tensor& p_vector1, Tensor& p_vector2);
	static void Concat(Tensor* p_result, Tensor* p_vector1, Tensor* p_vector2);

	void operator = (const Tensor& p_tensor);
	Tensor operator + (const Tensor& p_tensor) const;
	Tensor operator + (const double p_const) const;
	Tensor& operator += (const Tensor& p_tensor);
	Tensor& operator += (const double p_const);
	Tensor operator - (const Tensor& p_tensor) const;
	Tensor& operator -= (const Tensor& p_tensor);
	Tensor operator * (const Tensor& p_tensor) const;
	friend static Tensor operator * (const double p_const, const Tensor& p_tensor) { return p_tensor * p_const; }
	Tensor operator * (const double p_const) const;
	Tensor& operator *= (const double p_const);
	Tensor operator / (const double p_const) const;
	Tensor operator / (const Tensor& p_tensor) const;
	friend static Tensor operator / (const double p_const, const Tensor& p_tensor) {
		const Tensor temp(p_tensor);

		for (int i = 0; i < temp._size; i++) {
			temp._arr[i] = p_const / temp._arr[i];
		}

		return Tensor(temp);
	}
	Tensor& operator /= (const double p_const);
	double& operator [](const int p_index) const;

	void get_row(Tensor& p_tensor, int p_row) const;
	void get_column(Tensor& p_tensor, int p_column) const;

	Tensor T() const;
	Tensor diag() const;
	Tensor pow(double p_y) const;
	Tensor sqrt() const;
	Tensor dot(const Tensor& p_tensor) const;


	static Tensor apply(Tensor& p_source, double(*f)(double));
	static Tensor apply(Tensor& p_source1, Tensor& p_source2, double(*f)(double, double));

	int max_value_index() const;
	void override(Tensor* p_tensor) const;

	void fill(double p_value) const;

	double sum() const;

	int size() const { return _size; }
	int rank() const { return _rank; }
	int shape(const int p_index) const { return _shape[p_index]; }
	int* shape() const { return _shape; }

	double at(int p_x) const;
	double at(int p_y, int p_x) const;
	double at(int p_z, int p_y, int p_x) const;
	void set(int p_x, double p_val) const;
	void set(int p_y, int p_x, double p_val) const;
	void set(int p_z, int p_y, int p_x, double p_val) const;
	void inc(int p_x, double p_val) const;
	void inc(int p_x, int p_y, double p_val) const;
	void dec(int p_x, double p_val) const;
	void dec(int p_x, int p_y, double p_val) const;

	static double* alloc_arr(int p_size);
	static int* alloc_shape(int p_size);
	static int* copy_shape(int p_rank, int* p_shape);

	static double ew_dot(double p_x, double p_y);
	static double ew_div(double p_x, double p_y);
	static double ew_pow2(double p_x);
	static double ew_sqrt(double p_x);
	static double ew_abs(double p_x);
	static double sgn(double p_x);

	static int control;

private:
	void free_arr() const;
	void free_shape() const;
	
	void init_shape(int p_rank, int* p_shape, bool p_copy_shape);
	void init_shape(initializer_list<int> p_shape);
	void fill(INIT p_init, double p_value) const;
	
	double *_arr;
	int _rank{};
	int *_shape{};
	int _size{};
};

}
