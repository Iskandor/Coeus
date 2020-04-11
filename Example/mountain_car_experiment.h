#pragma once
#include "simple_continuous_env.h"
#include "mountain_car.h"

class mountain_car_experiment
{
public:
	mountain_car_experiment();
	~mountain_car_experiment();

	void test_simple(int p_epochs);
	void run_ddpg(int p_epochs);

private:
	simple_continuous_env _simple_env;
	mountain_car _env;
};

