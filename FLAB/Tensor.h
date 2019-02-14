#pragma once
#include <initializer_list>
#include <ostream>
#include <vector>

using namespace std;

extern "C" void daxpy_(int *N, double *DA, double *DX, int *INCX, double *DY, int *INCY);
extern "C" void dscal_(int *N, double *DA, double *DX, int *INCX);
extern "C" void dgemv_(char* T, int* M, int *N, double *DA, double *A, int *LDA, double *X, int *INCX, double *DY, double *Y, int *INCY);
extern "C" void dger_(int* M, int *N, double *DA, double *DX, int *INCX, double *DY, int *INCY, double *A, int *LDA);
extern "C" void dgemm_(char* TA, char* TB, int* M, int *N, int *K, double *DA, double *A, int *LDA, double *B, int *LDB, double *DC, double *C, int *LDC);

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
	Tensor(initializer_list<int> p_shape, INIT p_init, double p_value = 1.);
	Tensor(initializer_list<int> p_shape, double* p_data);
	Tensor(int p_rank, int* p_shape, double* p_data);
	Tensor(int p_rank, int* p_shape, INIT p_init, double p_value = 1.);
	Tensor(initializer_list<int> p_shape, initializer_list <double> p_inputs);
	Tensor(const Tensor &p_copy);
	~Tensor();
	Tensor operator-() const;
	
	static Tensor Zero(initializer_list<int> p_shape);
	static Tensor Ones(initializer_list<int> p_shape);
	static Tensor Value(initializer_list<int> p_shape, double p_value);
	static Tensor Random(initializer_list<int> p_shape, double p_limit);
	static Tensor concat(Tensor& p_vector1, Tensor& p_vector2);
	static void concat(Tensor* p_result, Tensor* p_vector1, Tensor* p_vector2);
	static Tensor concat(vector<Tensor>& p_vector);

	void operator = (const Tensor& p_tensor);
	Tensor operator + (const Tensor& p_tensor) const;
	Tensor operator + (double p_const) const;
	Tensor& operator += (const Tensor& p_tensor);
	Tensor& operator += (double p_const);
	Tensor operator - (const Tensor& p_tensor) const;
	Tensor& operator -= (const Tensor& p_tensor);
	Tensor operator * (const Tensor& p_tensor) const;
	friend static Tensor operator * (const double p_const, const Tensor& p_tensor) { return p_tensor * p_const; }
	Tensor operator * (double p_const) const;
	Tensor& operator *= (const Tensor& p_tensor);
	Tensor& operator *= (double p_const);
	Tensor operator / (double p_const) const;
	Tensor operator / (const Tensor& p_tensor) const;
	friend static Tensor operator / (const double p_const, const Tensor& p_tensor) {
		const Tensor temp(p_tensor);

		for (int i = 0; i < temp._size; i++) {
			temp._arr[i] = p_const / temp._arr[i];
		}

		return Tensor(temp);
	}
	Tensor& operator /= (double p_const);
	double& operator [](int p_index) const;

	void get_row(Tensor& p_tensor, int p_row) const;
	void set_row(Tensor& p_tensor, int p_row) const;
	void get_column(Tensor& p_tensor, int p_column) const;
	void set_column(Tensor& p_tensor, int p_column) const;

	Tensor T() const;
	Tensor pow(double p_y) const;
	Tensor sqrt() const;
	Tensor dot(const Tensor& p_tensor) const;
	Tensor outer_prod(const Tensor& p_tensor) const;


	static Tensor apply(Tensor& p_source, double(*f)(double));
	static Tensor apply(Tensor& p_source1, Tensor& p_source2, double(*f)(double, double));

	int max_value_index() const;
	void override(Tensor* p_tensor) const;
	void override(const double* p_data) const;
	void override(const int* p_data) const;

	void fill(double p_value) const;

	double sum() const;

	int size() const { return _size; }
	int rank() const { return _rank; }
	int shape(const int p_index) const { return _shape[p_index]; }
	int* shape() const { return _shape; }
	double* arr() const { return _arr; }

	double at(int p_x) const;
	double at(int p_y, int p_x) const;
	double at(int p_z, int p_y, int p_x) const;
	void set(int p_x, double p_val) const;
	void set(int p_y, int p_x, double p_val) const;
	void set(int p_z, int p_y, int p_x, double p_val) const;
	void inc(int p_x, double p_val) const;
	void inc(int p_x, int p_y, double p_val) const;

	static double* alloc_arr(int p_size);
	static int* alloc_shape(int p_size);
	static int* copy_shape(int p_rank, int* p_shape);

	static double ew_dot(double p_x, double p_y);
	static double ew_div(double p_x, double p_y);
	static double ew_pow2(double p_x);
	static double ew_sqrt(double p_x);
	static double ew_abs(double p_x);
	static double sgn(double p_x);

	static double dist(Tensor* p_tensor1, Tensor* p_tensor2);
	
	friend ostream &operator<<(ostream &output, const Tensor &p_tensor) {
		if (p_tensor.rank() == 1)
		{
			print_vector(output, p_tensor, false);
		}
		if (p_tensor.rank() == 2)
		{
			print_matrix(output, p_tensor, false);
		}

		return output;
	}

	void print(ostream &output) const;

private:

	static void print_vector(ostream &output, const Tensor &p_tensor, bool p_cm);
	static void print_matrix(ostream &output, const Tensor &p_tensor, bool p_cm);

	static double* ew_prod(const Tensor *p_x, const Tensor *p_y); 
	static double* mat_vec(const Tensor *p_A, const Tensor *p_x);

	void free_arr() const;
	void free_shape() const;
	
	void init_shape(int p_rank, int* p_shape, bool p_copy_shape);
	void init_shape(initializer_list<int> p_shape);
	void init_arr(double* p_arr);
	void fill(INIT p_init, double p_value) const;
	
	double *_arr;
	int _rank{};
	int *_shape{};
	int _size{};
};

}
