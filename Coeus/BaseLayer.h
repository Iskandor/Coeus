#pragma once
#include <string>
#include "Coeus.h"
#include "ParamModel.h"
#include "json.hpp"


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
			CONV = 8
		};

		BaseLayer(const string& p_id, int p_dim, initializer_list<int> p_in_dim);
		BaseLayer(json p_data);
		virtual ~BaseLayer();
		virtual BaseLayer* clone() = 0;

		virtual void init(vector<BaseLayer*>& p_input_layers);
		virtual void integrate(Tensor* p_input);
		virtual void activate() = 0;

		virtual void calc_derivative(map<string, Tensor*>& p_derivative) = 0;
		virtual void calc_gradient(map<string, Tensor>& p_gradient_map, map<string, Tensor*>& p_delta_map, map<string, Tensor*>& p_derivative_map) = 0;

		virtual void override(BaseLayer* p_source) = 0;
		virtual void reset() = 0;

		TYPE	get_type() const { return _type; }
		string	get_id() const { return _id; }

		bool is_valid() const { return _valid; }
		void set_valid(const bool p_val) { _valid = p_val; }

		Tensor* get_output() const { return _output; }
		int get_dim() const { return _dim; }
		virtual Tensor* get_dim_tensor() = 0;
		int get_in_dim() const { return _input_dim; }
		Tensor* get_in_dim_tensor() const { return _in_dim_tensor; }

		virtual json get_json() const;

	protected:
		
		int sum_input_dim(initializer_list<int> p_in_dim) const;
		explicit BaseLayer(BaseLayer* p_source);

		string		_id;
		TYPE		_type;
		int			_dim;
		Tensor*		_dim_tensor;
		int			_in_dim;
		Tensor*		_in_dim_tensor;
		int			_input_dim;

		int			_batch_size;
		bool		_batch{};
		Tensor*		_input;
		Tensor*		_output;

		vector<BaseLayer*>		_input_layer;
	private:
		bool	_valid;
	};
}