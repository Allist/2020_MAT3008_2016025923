/*
 * 2016025923 Jeon Yun Seong
 * HYU Numerical Analysis HW 2
 * Problem: 5x^4 - 22.4x^3 + 15.85272x^2 + 24.161472x - 23.4824832=0
 * Find the min locations using Newton method
 * - Use 1st and 2nd derivatives
 * Compare two methods
 * - 1. using exact 1st and 2nd derivatives
 * - 2. using approximation in page 18.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <float.h>

using namespace std;

double coefficients[5] = {-23.4824832, 24.161472, 15.85272, -22.4, 5};
// [0]: constants, [1]: x, [2]: x^2 ...
double prime[4];
double twoPrime[3];

void UsingExactDerivatives(void);
void UsingApproximation(void);
double CalcEquation(double x, int diffCnt);
void DifferenciateFunc(void);

int main(void)
{
    srand((unsigned int)time(NULL));
    DifferenciateFunc();

    UsingExactDerivatives();
    UsingApproximation();

    return 0;
}

void UsingExactDerivatives(void)
{
    double interval = 0.00001;
    double minFX = DBL_MAX;
    double minX = DBL_MAX;

    // Iterate 5 times using random number.
    for(int i = 0; i < 5; ++i) {
        double x = rand()%0x0FFF;
        if(rand()%2) { // determine plus or minus
            x *= -1.0;
        }

        double nextX;
        double errorX = abs(x * 100);
        
        while(interval < errorX){
            nextX = x - CalcEquation(x, 1) / CalcEquation(x, 2);
            errorX = abs((nextX-x)/nextX);
            x = nextX;
        }
        cout << "min : " << x << ", f(min) : " << CalcEquation(x, 0) << endl;
        cout << "errorX : " << abs(errorX) << endl;
        if(CalcEquation(x, 0) < minFX) {
            minFX = CalcEquation(x, 0);
            minX = x;
        }
    }
    cout << "Guess min: " << minX << ", f(x) : " << minFX << endl;
}

void UsingApproximation(void)
{
    double interval = 0.001;
    double minFX = DBL_MAX;
    double minX = DBL_MAX;

    // Iterate 5 times using random number.
    for(int i = 0; i < 5; ++i) {
        double x = rand()%0x0FFF;
        if(rand()%2) { // determine plus or minus
            x *= -1.0;
        }

        double nextX;
        double errorX = abs(x * 100);

        while(interval < errorX){
            double fx = CalcEquation(x, 0);
            double fx1 = CalcEquation(x + interval, 0);
            double fx_1 = CalcEquation(x - interval, 0);
            double firstDiff = (fx1 - fx) / interval;
            double secondDiff = (fx1 - 2*fx + fx_1) / (interval * interval);

            nextX = x - firstDiff/secondDiff;
            errorX = abs((nextX-x)/nextX);
            x = nextX;
        }
        cout << "min : " << x << ", f(min) : " << CalcEquation(x, 0) << endl;
        cout << "errorX : " << errorX << endl;
        if(CalcEquation(x, 0) < minFX) {
            minFX = CalcEquation(x, 0);
            minX = x;
        }
    }
    cout << "Guess min: " << minX << ", f(x) : " << minFX << endl;
}

double CalcEquation(double x, int diffCnt)
{
    double ret = 0;

    switch(diffCnt) {
        case 0:
            for(int i = 0; i < 5; ++i) {
                ret += pow(x, i) * coefficients[i];
            }
            break;
        case 1:
            for(int i = 0; i < 4; ++i) {
                ret += pow(x, i) * prime[i];
            }
            break;
        case 2:
            for(int i = 0; i < 3; ++i) {
                ret += pow(x, i) * twoPrime[i];
            }
    }

    return ret;
}

void DifferenciateFunc(void)
{
    for(int i = 1; i < 5; ++i) {
        prime[i-1] = coefficients[i]*i;
    }
    for(int i = 1; i < 4; ++i) {
        twoPrime[i-1] = prime[i]*i;
    }
}