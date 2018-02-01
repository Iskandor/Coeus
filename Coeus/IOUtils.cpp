#include "IOUtils.h"
#include <fstream>

using namespace Coeus;

IOUtils::IOUtils()
{
}

IOUtils::~IOUtils()
{
}

json IOUtils::save_layer(BaseLayer* p_layer) {
	json data;

	data["_header"] = "Coeus";
	data["_type"] = p_layer->type();

	switch (p_layer->type()) {
	case BaseLayer::SOM:
		data["_network"] = write_som(static_cast<SOM*>(p_layer));
		break;
	case BaseLayer::MSOM:
		data["_network"] = write_msom(static_cast<MSOM*>(p_layer));
		break;
	default:;
	}

	return data;
}

BaseLayer* IOUtils::load_layer(json p_data) {
	BaseLayer* result = nullptr;

	const BaseLayer::TYPE type = static_cast<BaseLayer::TYPE>(p_data["_type"].get<int>());

	switch (type) {
	case BaseLayer::SOM:
		result = read_som(p_data["_network"]);
		break;
	case BaseLayer::MSOM:
		result = read_msom(p_data["_network"]);
		break;
	default:;
	}

	return result;
}

void IOUtils::save_network(const string p_filename, BaseLayer* p_layer) {
	json data = save_layer(p_layer);

	ofstream file;
	file.open(p_filename);
	file << data.dump();
	file.close();
}

BaseLayer* IOUtils::load_network(const string p_filename) {
	BaseLayer* result = nullptr;

	json data;
	ifstream file;

	try {
		file.open(p_filename);
	}
	catch (std::ios_base::failure& e) {
		std::cerr << e.what() << '\n';
	}

	if (file.is_open()) {
		file >> data;
		
		result = load_layer(data);
	}
	file.close();

	return result;
}

json IOUtils::write_som(SOM* p_som) {
	json result;

	result["dim_x"] = p_som->dim_x();
	result["dim_y"] = p_som->dim_y();
	result["groups"]["input"] = write_neural_group(p_som->get_input_group());
	result["groups"]["lattice"] = write_neural_group(p_som->get_lattice());
	result["connections"]["input_lattice"] = write_connection(p_som->get_input_lattice());

	return result;
}

json IOUtils::write_msom(MSOM* p_msom) {
	json result = write_som(p_msom);

	result["alpha"] = p_msom->get_alpha();
	result["beta"] = p_msom->get_beta();
	result["groups"]["context"] = write_neural_group(p_msom->get_context_group());
	result["connections"]["context_lattice"] = write_connection(p_msom->get_context_lattice());

	return result;
}

SOM* IOUtils::read_som(const json p_data) {
	return new SOM(p_data);
}

MSOM* IOUtils::read_msom(const json p_data) {
	return new MSOM(p_data);
}

json IOUtils::write_neural_group(NeuralGroup* p_group) {
	return json({ { "id", p_group->getId() }, { "dim", p_group->getDim() }, { "actfn", p_group->getActivationFunction() } });
}

json IOUtils::write_connection(Connection* p_connection) {
	json result;

	result["id"] = p_connection->get_id();
	result["in_id"] = p_connection->get_in_id();
	result["out_id"] = p_connection->get_out_id();
	result["in_dim"] = p_connection->get_in_dim();
	result["out_dim"] = p_connection->get_out_dim();

	stringstream ss;

	for (int i = 0; i < p_connection->get_weights()->size(); i++) {
		double w = p_connection->get_weights()->at(i);
		ss.write((char*)&w, sizeof(double));		
	}

	result["weights"] = ss.str();

	return result;
}

NeuralGroup* IOUtils::read_neural_group(const json p_data) {
	return new NeuralGroup(p_data);
}

Connection* IOUtils::read_connection(const json p_data) {
	return new Connection(p_data);
}
