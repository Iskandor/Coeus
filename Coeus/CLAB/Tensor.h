#pragma once
#include <initializer_list>
#include <ostream>
#include <vector>

using namespace std;

class __declspec(dllexport) Tensor
{
public:
	enum INIT {
		ZERO = 0,
		ONES = 1,
		VALUE = 2,
		RANDOM = 3,
		INCREMENTAL = 4
	};

	Tensor();
	Tensor(initializer_list<int> p_shape, INIT p_init, float p_value = 1.);
	Tensor(initializer_list<int> p_shape, float* p_data);
	Tensor(initializer_list<int> p_shape, const vector<uint8_t>& p_data);
	Tensor(int p_rank, int* p_shape, float* p_data);	
	Tensor(int p_rank, int* p_shape, INIT p_init, float p_value = 1.);
	Tensor(initializer_list<int> p_shape, initializer_list <float> p_inputs);
	Tensor(const Tensor &p_copy);
	~Tensor();
	Tensor operator-() const;
	

	static Tensor Zero( initializer_list<int> p_shape);
	static Tensor Ones( initializer_list<int> p_shape);
	static Tensor Value( initializer_list<int> p_shape, float p_value);
	static Tensor Random( initializer_list<int> p_shape, float p_limit);

	Tensor& operator = (const Tensor& p_copy);
	float& operator []( int p_index) const;
	Tensor T();
	Tensor vec() const;
	Tensor inv() const;
	Tensor pinv() const;

	Tensor& operator += (const Tensor& p_rhs);
	Tensor& operator += (float p_rhs);
	Tensor& operator -= (const Tensor& p_rhs);
	Tensor& operator -= (float p_rhs);
	Tensor operator *= (const Tensor& p_rhs) const;
	Tensor& operator *= (float p_rhs);
	Tensor& operator /= (float p_rhs);

	Tensor operator + (const Tensor& p_rhs) const;
	Tensor operator + (float p_rhs) const;
	Tensor operator - (const Tensor& p_rhs) const;
	Tensor operator - (float p_rhs) const;
	Tensor operator * (const Tensor& p_rhs) const;
	Tensor operator * (float p_rhs) const;
	Tensor operator / (float p_rhs) const;
	friend Tensor operator * (float p_lhs, const Tensor& p_rhs);

	float dot(const Tensor& p_rhs) const;
	Tensor& pow (float p_y);
	Tensor& sqrt(float p_y);

	void get_row(Tensor& p_tensor, int p_row) const;
	void set_row(Tensor& p_tensor, int p_row) const;
	void get_column(Tensor& p_tensor, int p_column) const;
	void set_column(Tensor& p_tensor, int p_column) const;
	
	float max_value() const;
	int max_value_index() const;
	int count_value(float p_value) const;

	void override(Tensor* p_tensor) const;
	void override(const float* p_data) const;
	void override(const int* p_data) const;

	void fill(float p_value) const;

	int size() const { return _size; }
	int rank() const { return _rank; }
	int shape(const int p_index) const { return _shape[p_index]; }
	int* shape() const { return _shape; }
	float* arr() const { return _arr; }

	float at(int p_x) const;
	float at(int p_y, int p_x) const;
	float at(int p_z, int p_y, int p_x) const;
	float at(int p_n, int p_z, int p_y, int p_x) const;
	void set(int p_x, float p_val) const;
	void set(int p_y, int p_x, float p_val) const;
	void set(int p_z, int p_y, int p_x, float p_val) const;
	void set(int p_n, int p_z, int p_y, int p_x, float p_val) const;

	float element_prod() const;
	float trace() const;

	static float* alloc_arr(int p_size);
	static int* alloc_shape(int p_size);
	static int* copy_shape(int p_rank, const int* p_shape);

	void push_back(Tensor* p_tensor);
	void push_back(Tensor* p_tensor, int p_y, int p_x, int p_size);
	void push_back(float p_value);
	void insert_row(Tensor* p_tensor);
	void insert_column(Tensor* p_tensor);
	
	void splice(int p_start, Tensor* p_output) const;
	void reset_index();
	static Tensor*	concat(vector<Tensor*> &p_input);
	static Tensor	concat(vector<Tensor>& p_vector);
	static void padding(Tensor& p_dest, Tensor& p_source, int p_padding);

	void padding(int p_padding);
	void reshape(initializer_list<int> p_shape);
	Tensor slice(int p_index) const;
	void replicate(int p_n);

	Tensor avg_sum(int p_dim);

	bool has_NaN_Inf() const;

	static void subregion(Tensor* p_dest, Tensor* p_source, int p_y, int p_x, int p_h, int p_w);
	static void subregion(Tensor* p_dest, Tensor* p_source, int p_z, int p_y, int p_x, int p_h, int p_w);
	static void subregion(Tensor* p_dest, Tensor* p_source, int p_n, int p_z, int p_y, int p_x, int p_h, int p_w);
	static void add_subregion(Tensor* p_dest, int p_yd, int p_xd, Tensor* p_source, int p_y, int p_x, int p_h, int p_w);
	static void add_subregion(Tensor* p_dest, int p_zd, int p_yd, int p_xd, int p_hd, int p_wd, Tensor* p_source, int p_y, int p_x, int p_h, int p_w);
	static void add_subregion(Tensor* p_dest, int p_nd, int p_zd, int p_yd, int p_xd, int p_hd, int p_wd, Tensor* p_source, int p_y, int p_x, int p_h, int p_w);
	static int subregion_max_index(Tensor* p_source, int p_y, int p_x, int p_h, int p_w);

	static void slice(Tensor* p_dest, Tensor* p_source, int p_index);

	static void im2col(Tensor* p_image, Tensor* p_column, int p_extent, int p_padding, int p_stride);
	static void col2im(Tensor* p_column, Tensor* p_image, int p_extent, int p_padding, int p_stride);


	static int kronecker_delta(const int i, const int j) {
		return i == j ? 1 : 0;
	}

	static int sign(const float x)
	{
		int result = 0;

		if (x > 0) result = 1;
		if (x < 0) result = -1;
		return result;
	}

	static float max(float a, float b)
	{
		return a > b ? a : b;
	}

	friend ostream &operator<<(ostream &output, const Tensor &p_tensor) {
		if (p_tensor.rank() == 1)
		{
			print_vector(output, p_tensor, false);
		}
		if (p_tensor.rank() == 2)
		{
			print_matrix(output, p_tensor, false);
		}
		if (p_tensor.rank() == 3)
		{
			print_volume(output, p_tensor);
		}
		if (p_tensor.rank() == 4)
		{
			print_batch_volume(output, p_tensor);
		}
		return output;
	}

	static void enable_pooling(bool p_value);
	static bool is_enabled_pooling();

private:	
	static bool _pooling;
	
	static void print_vector(ostream &output, const Tensor &p_tensor, bool p_cm);
	static void print_matrix(ostream &output, const Tensor &p_tensor, bool p_cm);
	static void print_volume(ostream &output, const Tensor &p_tensor);
	static void print_batch_volume(ostream &output, const Tensor &p_tensor);

	void free_arr() const;
	void free_shape() const;
	
	void init_shape(int p_rank, int* p_shape, bool p_copy_shape);
	void init_shape(initializer_list<int> p_shape);
	void fill(INIT p_init, float p_value) const;

	void check_size_gt(int p_size) const;
	void check_size_eq(int p_size) const;
	void check_rank_gt(int p_rank) const;
	void check_rank_eq(int p_rank) const;
	void check_square() const;
	
	float *_arr;
	int _rank;
	int *_shape;
	int _size;
	int _end;
	bool _transpose;
};
