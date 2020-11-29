/*
 * 2016025923 Jeon Yun Seong
 * HYU Numerical Analysis HW 1
 * Problem: 5x^4 - 22.4x^3 + 15.85272x^2 + 24.161472x - 23.4824832=0
 * Find the roots of the polynomial equation
 * - Use bisection and Newton-Raphson, respectively
*/

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double coefficients[5] = {-23.4824832, 24.161472, 15.85272, -22.4, 5};
// [0]: constants, [1]: x, [2]: x^2 ...
double prime[4];

void Bisection(void);
void NewtonRaphson(void);
vector<pair<double, double>> FindBrackets(double lower, double upper, double gap);
double FindRootByBisection(double lower, double upper);
double FindRootByNewtonRaphson(double sample);
double CalcEquation(double x);
void DifferenciateFunc(void);
double CalcPrimeEquation(double x);


int main(void)
{
    cout << "Numerical Analysis HW 1" << endl;
    Bisection();
    NewtonRaphson();

    return 0;
}

void Bisection(void)
{
    vector<pair<double, double>> brackets;
    vector<pair<double, double>>::iterator bracket_it;
    vector<double> roots;
    double min, max, gap;

    cout << "Start Bisection" << endl;

    min = -2;
    max = 4;
    gap = 0.01;

    cout << "Range: " << min << ", " << max << endl;
    cout << "Gap: " << gap << endl;

    brackets = FindBrackets(min, max, gap);

    // If roots found, filtering....
    bracket_it = brackets.begin();
    while(bracket_it != brackets.end()) {
        //cout << bracket_it->first << ", " << bracket_it->second << endl;// For debug
        if(bracket_it->first == bracket_it->second) {
            cout << "Root : " << bracket_it->first << endl;
            roots.push_back(bracket_it->first);
            bracket_it = brackets.erase(bracket_it);
        } else {
            bracket_it++;
        }
    }

    bracket_it = brackets.begin();
    while(bracket_it != brackets.end()) {
        cout << "Root : " << FindRootByBisection(bracket_it->first, bracket_it->second) << endl;
        bracket_it++;
    }
}

void NewtonRaphson(void)
{
    double samples[4] = {-2, 0, 2,  4};

    cout << "Start NewtonRaphson" << endl;

    DifferenciateFunc();

    for(int i = 0; i < 4; ++i) {
        cout << "Root : " << FindRootByNewtonRaphson(samples[i]) << endl;
    }
}

vector<pair<double, double>> FindBrackets(double lower, double upper, double gap)
{
    vector<pair<double, double>> brackets;
    for(double x = lower; x < upper; x += gap) {
        int sign_left, sign_right;

        double left = CalcEquation(x);
        if(left == 0) {
            brackets.push_back(make_pair(x, x));
            continue;
        } else if (left < 0) {
            sign_left = -1;
        } else {
            sign_left = 1;
        }

        double right = CalcEquation(x+gap);
        if(right == 0) {
            brackets.push_back(make_pair(x+gap, x+gap));
            x += gap;
            continue;
        } else if (right < 0) {
            sign_right = -1;
        } else {
            sign_right = 1;
        }

        if(sign_left * sign_right == -1) {
            brackets.push_back(make_pair(x, x+gap));
            //cout << x << ", " << x+gap << ", " << left << ", " << right << endl; // for Debug
        }
    }

    return brackets;
}

double FindRootByBisection(double lower, double upper)
{
    double minus, plus, left, right;
    double tmp = CalcEquation(lower);
    if(tmp < 0) {
        minus = tmp;
        plus = CalcEquation(upper);
        left = lower;
        right = upper;
    } else {
        plus = tmp;
        minus = CalcEquation(upper);
        left = upper;
        right = lower;
    }
    for(int i = 0; i < 10; ++i) {
        double center = (left + right)/2;
        double tmp = CalcEquation(center);
        if(tmp < 0) {
            left = center;
        } else if(tmp > 0){
            right = center;
        } else {
            return center;
        }
    }
    return (left + right)/2;
}

double FindRootByNewtonRaphson(double sample)
{
    for(int i = 0; i < 10; ++i) {
        sample = sample - CalcEquation(sample)/CalcPrimeEquation(sample);
    }

    return sample;
}

double CalcEquation(double x)
{
    double ret = 0;

    for(int i = 0; i < 5; ++i) {
        ret += pow(x, i) * coefficients[i];
    }

    return ret;
}

void DifferenciateFunc(void)
{
    for(int i = 1; i < 5; ++i) {
        prime[i-1] = coefficients[i]*i;
    }
}

double CalcPrimeEquation(double x)
{
    double ret = 0;

    for(int i = 0; i < 4; ++i) {
        ret += pow(x, i) * prime[i];
    }

    return ret;
}