#include "SOM_analyzer.h"

using namespace Coeus;

SOM_analyzer::SOM_analyzer()
{
	_q_error = 0;
}


SOM_analyzer::~SOM_analyzer()
{
}

void SOM_analyzer::update(SOM* p_som, const int p_winner)
{
	_winner_set.insert(p_winner);
	_q_error += p_som->calc_distance(p_winner);
}

void SOM_analyzer::end_epoch()
{
	_winner_set.clear();
	_q_error = 0;
}

void SOM_analyzer::merge(vector<SOM_analyzer*> &p_analyzers) {
	for (auto it = p_analyzers.begin(); it != p_analyzers.end(); ++it) {
		_q_error += (*it)->_q_error;
		_winner_set.insert((*it)->_winner_set.begin(), (*it)->_winner_set.end());
		(*it)->end_epoch();
	}
}

double SOM_analyzer::winner_diff(const int p_size) const {
	return static_cast<double>(_winner_set.size()) / static_cast<double>(p_size);
}
