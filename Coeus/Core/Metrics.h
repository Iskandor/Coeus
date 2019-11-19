#pragma once

namespace Coeus
{
	class __declspec(dllexport) Metrics
	{
	public:
		Metrics();
		~Metrics();

		static float euclidean_distance(int p_x1, int p_y1, int p_x2, int p_y2);
		static float gaussian_distance(float p_d, float p_sigma = 1, float p_mu = 0);
		static float binary_distance(float p_d, float p_h);

	};
}