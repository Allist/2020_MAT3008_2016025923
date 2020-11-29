#include <opencv2/core.hpp>
#include <iostream>
#include <random>
#include <vector>

const int START = -5;
const int END = 6;
const int CNT = END-START+1;

using namespace std;

void Make_Points(cv::Mat &points)
{
    random_device rd;
    mt19937_64 gen(rd());
    normal_distribution<double> nd(0.0, 2.0);
    for(int i = START; i <= END; ++i) {
        double x = i;
        double y = 2 * i - 1 + nd(gen);
        //double y = 2 * i - 1;
        cv::Vec2d point(x, y);
        cv::vconcat(points, point.t(), points);
    }
}

void Do_RANSAC(cv::Mat const &points)
{
    std::vector<int> order = {0,1,2,3,4,5,6,7,8,9,10,11};
    cv::Mat sample_points(6, 2, CV_64FC1);
    cv::Mat A(6, 2, CV_64FC1);
    cv::Mat b(6, 1, CV_64FC1);
    double optimal_dist = DBL_MAX;
    cv::Mat optimal_vec(1, 2, CV_64FC1);
    for(int i = 0; i < 30; ++i) {
        std::random_shuffle(order.begin(), order.end());
        for(int j = 0; j < 6; ++j) {
            sample_points.row(j) = points.row(j);
            points.row(j).copyTo(sample_points.row(j));
            A.at<double>(j, 0) = points.at<double>(order[j], 0);
            A.at<double>(j, 1) = 1;
            b.at<double>(j, 0) = points.at<double>(order[j], 1);
        }
        cv::Mat preX = (A.t() * A).inv() * A.t() * b; // {a, b}
        cv::Mat calcB = A*preX;

        double dist = cv::sum(cv::abs(calcB-b))[0];
        if(dist < optimal_dist) {
            optimal_dist = dist;
            preX.copyTo(optimal_vec);
        }
        if(i % 10 == 0) {
            cout << "try: " << i << endl;
            cout << optimal_dist << endl;
            cout << optimal_vec << endl;
        }
    }
}

void Do_Least_Square_FULL(cv::Mat const &points)
{
    cv::Mat A(12, 2, CV_64FC1);
    cv::Mat b(12, 1, CV_64FC1);
    for(int j = 0; j < 12; ++j) {
        A.at<double>(j, 0) = points.at<double>(j, 0);
        A.at<double>(j, 1) = 1;
        b.at<double>(j, 0) = points.at<double>(j, 1);
    }
    cv::Mat preX = (A.t() * A).inv() * A.t() * b; // {a, b}
    cv::Mat calcB = A*preX;
    double dist = cv::sum(cv::abs(calcB-b))[0];
    cout << "using all samples" << endl;
    cout << dist << endl;
    cout << preX << endl;
}

int main(void)
{
    cv::Mat points(0, 2, CV_64FC1);
    Make_Points(points);
    cout << points << endl;
    Do_RANSAC(points);
    Do_Least_Square_FULL(points);
}