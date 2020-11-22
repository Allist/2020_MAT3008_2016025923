#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>

using namespace std;

int main(void)
{
    cv::Mat origin_img;
    cv::Mat rgb_img;
    cv::Mat yuv_img;
    cv::Mat mul_ch[3]; // y u v
    cv::Mat colorTbl[2]; // GB GR
    cv::Mat print_ch[3]; // y GB GR
    cv::Mat mean, stddev;
    string dir = "./Images/";

    double correlation[20][6] = {0, };//BG BR GR YU YV UV

    // Make Look Up Table
    colorTbl[0] = cv::Mat(1, 256, CV_8UC3); // GB
    for(int i = 0; i < 256; ++i) {
        colorTbl[0].at<cv::Vec3b>(i)[0] = i; // B
        colorTbl[0].at<cv::Vec3b>(i)[1] = 255-i; // G
        colorTbl[0].at<cv::Vec3b>(i)[2] = 0; // R
    }
    colorTbl[1] = cv::Mat(1, 256, CV_8UC3); // GR
    for(int i = 0; i < 256; ++i) {
        colorTbl[1].at<cv::Vec3b>(i)[0] = 0; // B
        colorTbl[1].at<cv::Vec3b>(i)[1] = 255-i; // G
        colorTbl[1].at<cv::Vec3b>(i)[2] = i; // R
    }

    cout << "|-idx-|-----BG-----|-----BR-----|-----GR-----|-----YU-----|-----YV-----|-----UV-----|" << endl;

    for(int idx = 0; idx  < 10; ++idx) {
        string filename = to_string(idx);
        string extension = ".jpg";
        origin_img = cv::imread(dir+filename+extension);
        int pixels = origin_img.rows * origin_img.cols;
        origin_img.copyTo(rgb_img);
        cv::cvtColor(origin_img, yuv_img, cv::COLOR_BGR2YUV);// BGR to YUV 444

        // Calc BGR correlation coefficients
        rgb_img.convertTo(rgb_img, CV_64FC3);
        cv::meanStdDev(rgb_img, mean, stddev); // bgr format
        cv::split(rgb_img, mul_ch);
        for(int i = 0; i < 3; ++i) {
            double mean1 = mean.at<cv::Vec3f>(0)[i/2]; // 0, 0, 1
            double mean2 = mean.at<cv::Vec3f>(0)[(i+3)/2]; // 1, 2, 2
            double stddev1 = stddev.at<cv::Vec3f>(0)[i/2];
            double stddev2 = stddev.at<cv::Vec3f>(0)[(i+3)/2];
            double cov = (mul_ch[i/2]-mean1).dot(mul_ch[(i+3)/2]-mean2) / pixels;
            correlation[idx][i] = cov / (stddev1 * stddev2);
        }

        // Calc YUV correlation coefficients
        yuv_img.convertTo(yuv_img, CV_64FC3);
        cv::meanStdDev(yuv_img, mean, stddev);
        cv::split(yuv_img, mul_ch);
        for(int i = 0; i < 3; ++i) {
            double mean1 = mean.at<cv::Vec3f>(0)[i/2]; // 0, 0, 1
            double mean2 = mean.at<cv::Vec3f>(0)[(i+3)/2]; // 1, 2, 2
            double stddev1 = stddev.at<cv::Vec3f>(0)[i/2];
            double stddev2 = stddev.at<cv::Vec3f>(0)[(i+3)/2];
            double cov = (mul_ch[i/2]-mean1).dot(mul_ch[(i+3)/2]-mean2) / pixels;
            correlation[idx][i+3] = cov / (stddev1 * stddev2);
        }

        // Print correlation coefficients
        cout << "|--" << idx << "--|";
        //printf("|--%d--|", idx);
        for(int i = 0; i < 6; ++i) {
            cout.width(12);
            cout << correlation[idx][i];
            cout << "|";
        }
        cout << endl;

        /* Image write YUV format */
        cv::cvtColor(origin_img, yuv_img, cv::COLOR_BGR2YUV);// BGR to YUV 444
        cv::split(yuv_img, mul_ch);
        mul_ch[0].copyTo(print_ch[0]);
        for(int i = 0; i < 2; ++i) {
            cv::cvtColor(mul_ch[i+1], mul_ch[i+1], cv::COLOR_GRAY2BGR);
            cv::LUT(mul_ch[i+1], colorTbl[i], print_ch[i+1]);
        }
        // Save Img
        for(int i = 0; i < 3; ++i) {
            cv::imwrite(dir+filename + "_" + to_string(i) + extension, print_ch[i]);
        }
        /* print
        for(int i = 0; i < 3; ++i) {
            cv::namedWindow("2D DFT image", cv::WINDOW_AUTOSIZE);
            cv::imshow("2D DFT image", print_ch[i]);
            cv::waitKey(0);
        }
        */
    }

    return 0;
}