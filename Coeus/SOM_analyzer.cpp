#include "SOM_analyzer.h"

using namespace Coeus;

SOM_analyzer::SOM_analyzer(SOM* p_som)
{
	_som = p_som;
	_q_error = 0;
}


SOM_analyzer::~SOM_analyzer()
{
}

void SOM_analyzer::update(const int p_winner)
{
	_winner_set.insert(p_winner);
	_q_error += _som->calc_distance(p_winner);
}

void SOM_analyzer::end_epoch()
{
	_winner_set.clear();
	_q_error = 0;
}

double SOM_analyzer::winner_diff()
{
	return (double)_winner_set.size() / (double)_som->get_lattice()->getDim();
}
