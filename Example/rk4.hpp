
class RK4 {
public:
    RK4() {};
    ~RK4() {};

    static double rk4 ( double t0, double u0, double dt, double f ( double t, double u ) );
    static double *rk4vec ( double t0, int n, double u0[], double p[], double dt, double *f ( double t, int n, double u[], double params[] ) );

private:

};

