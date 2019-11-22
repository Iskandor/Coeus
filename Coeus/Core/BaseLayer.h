#pragma once
#include <string>
#include "Coeus.h"
#include "ParamModel.h"
#include "json.hpp"
#include <stack>
#include "Gradient.h"


using namespace std;
using namespace nlohmann;

namespace Coeus
{
	class __declspec(dllexport) BaseLayer : public ParamModel
	{
	public:
		enum TYPE
		{
			SOM = 1,
			MSOM = 2,
			CORE = 4,
			RECURRENT = 5,
			LSTM = 6,
			LSOM = 7,
			CONV = 8,
			GRU = 9
		};

		BaseLayer(const string& p_id, int p_dim, initializer_list<int> p_in_dim);
		BaseLayer(json p_data);
		virtual ~BaseLayer();
		virtual BaseLayer* copy(bool p_clone) { return nullptr; }

		virtual void init(vector<BaseLayer*>& p_input_layers, vector<BaseLayer*>& p_output_layers);
		virtual void integrate(Tensor* p_input);
		virtual void activate() = 0;

		virtual void calc_derivative(map<string, Tensor*>& p_derivative) = 0;
		virtual void calc_gradient(Gradient& p_gradient_map, map<string, Tensor*>& p_derivative_map);
		
		virtual void reset() = 0;

		TYPE	get_type() const { return _type; }
		string	get_id() const override { return _id; }

		bool is_valid() const { return _valid; }
		void set_valid(const bool p_val) { _valid = p_val; }

		Tensor* get_output() const { return _output; }
		int get_dim() const { return _dim; }
		virtual Tensor* get_dim_tensor() = 0;
		int get_in_dim() const { return _input_dim; }
		Tensor* get_in_dim_tensor() const { return _in_dim_tensor; }
		bool is_recurrent() const { return _is_recurrent; }
		void set_mode(const RECURRENT_MODE p_value) { _mode = p_value; }

		Tensor* get_delta_in(const string& p_id);
		void set_delta_out(Tensor* p_value);

		virtual json get_json() const;

		vector<string> unfold_layer();

	protected:
		
		int sum_input_dim(initializer_list<int> p_in_dim) const;

		string		_id;
		TYPE		_type;
		int			_dim;
		Tensor*		_dim_tensor;
		int			_in_dim;
		Tensor*		_in_dim_tensor;
		int			_input_dim;

		int			_batch_size;
		bool		_batch;
		Tensor*		_input;
		Tensor*		_output;

		Tensor*				 _delta_out;
		map<string, Tensor*> _delta;
		map<string, Tensor*> _delta_in;

		bool		_is_recurrent;
		RECURRENT_MODE	_mode;
		stack<map<string, Tensor*>> _bptt_values;

		vector<BaseLayer*>		_input_layer;
		vector<BaseLayer*>		_output_layer;
	private:
		bool	_valid;
	};
}