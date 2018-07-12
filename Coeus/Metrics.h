#pragma once

namespace Coeus
{
	class __declspec(dllexport) Metrics
	{
	public:
		Metrics();
		~Metrics();

		static double euclidean_distance(int p_x1, int p_y1, int p_x2, int p_y2);
		static double gaussian_distance(double p_d, double p_sigma = 1, double p_mu = 0);
		static double binary_distance(double p_d, double p_h);

	};
}