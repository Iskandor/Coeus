
class RK4 {
public:
    RK4() {};
    ~RK4() {};

    static float rk4 ( float t0, float u0, float dt, float f ( float t, float u ) );
    static float *rk4vec ( float t0, int n, float u0[], float p[], float dt, float *f ( float t, int n, float u[], float params[] ) );

private:

};

