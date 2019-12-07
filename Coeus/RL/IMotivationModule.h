#pragma once

namespace Coeus
{
	class __declspec(dllexport) IMotivationModule
	{
	public:
		IMotivationModule();
		~IMotivationModule();

		virtual float uncertainty_motivation() = 0;
		virtual float familiarity_motivation() = 0;
		virtual float intermediate_novelty_motivation(float p_sigma) = 0;
		virtual float surprise_motivation() = 0;

		virtual float progress_uncertainty_motivation() = 0;
		virtual float progress_familiarity_motivation() = 0;
	};
}