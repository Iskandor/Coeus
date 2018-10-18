#pragma once
namespace Coeus
{
	class __declspec(dllexport) LearningRateModule
	{
	public:
		LearningRateModule();
		~LearningRateModule();

		void init(double p_alpha_min, double p_alpha_max, int p_T0, int p_Tmult);
		double update();

	private:
		double _alpha_min;
		double _alpha_max;
		int _Tcur;
		int _Ti;
		int _Tmult;
	};
}