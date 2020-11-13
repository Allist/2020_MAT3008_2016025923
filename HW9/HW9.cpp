#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

cv::Mat ORIGIN_IMG;
cv::Mat BGR_SPLIT_IMG[3];
cv::Mat IDCT_IMG;
cv::Mat MODIFY_IMG;
double ZERO_C = 1/sqrt(2);

class CoeffClass{
    public:
    double val;
    double abs_val;
    int x;
    int y;
    CoeffClass(double val, int x, int y) {
        this->val = val;
        this->abs_val = abs(val);
        this->x = x;
        this->y = y;
    }
    bool operator >(CoeffClass &CC) {
        return this->abs_val > CC.abs_val;
    }
    bool operator <(CoeffClass &CC) {
        return this->abs_val < CC.abs_val;
    }
};

bool CmpCoeff(CoeffClass a, CoeffClass b)
{
    return a > b;
}

bool LoadImage(string imgFilePath)
{
    ORIGIN_IMG = cv::imread(imgFilePath);

    if(ORIGIN_IMG.empty())
        return false;
    return true;
}

/*
 * S_C_IMG: single channel image
 */
void MY_DCT(int begin_x, int begin_y, cv::Mat &coeff, cv::Mat &S_C_IMG)
{
    cv::Mat block = S_C_IMG(cv::Rect(begin_x, begin_y, 16, 16));
    for(int v = 0; v < 16; ++v) { // related with y
        double vPi = M_PIf64 * v;
        for(int u = 0; u < 16; ++u) { // related with x
            double uPi = M_PIf64 * u;

            double sum = 0.0;
            for(int x = 0; x < 16; ++x) {
                for(int y = 0; y < 16; ++y) {
                    double val = 1.0;
                    val *= block.at<unsigned char>(cv::Point(x, y));
                    val *= cosf64(vPi * (2*y+1) / 32);
                    val *= cosf64(uPi * (2*x+1) / 32);
                    sum += val;
                }
            }
            sum = sum / 8;
            if(u == 0) sum *= ZERO_C;
            if(v == 0) sum *= ZERO_C;
            coeff.at<double>(cv::Point(u, v)) = sum;
        }
    }
}

void Select16Coeff(cv::Mat &coeff)
{
    vector<CoeffClass> sortArr;
    for(int x = 0; x < 16; ++x) {
        for(int y = 0; y < 16; ++y) {
            sortArr.push_back(CoeffClass(coeff.at<double>(cv::Point(x, y)), x, y));
        }
    }
    sort(sortArr.begin(), sortArr.end(), CmpCoeff);
    coeff.~Mat();
    coeff = cv::Mat::zeros(cv::Size(16, 16), CV_64FC1);

    int i = 0;
    vector<CoeffClass>::iterator it = sortArr.begin();
    for( ; i < 16; ++it, ++i) {
            coeff.at<double>(cv::Point(it->x, it->y)) = it->val;
    }
}

void MY_IDCT(int begin_x, int begin_y, cv::Mat &coeff)
{
    for(int x = 0; x < 16; ++x) {
        for(int y = 0; y < 16; ++y) {
            double sum = 0.0;
            for(int v = 0; v < 16; ++v) { // related with y
                for(int u = 0; u < 16; ++u) { // related with x
                    double val = 1.0;
                    if(u == 0) val *= ZERO_C;
                    if(v == 0) val *= ZERO_C;
                    val *= coeff.at<double>(cv::Point(u, v));
                    val *= cosf64(M_PIf64 * v * (2*y + 1) / 32);
                    val *= cosf64(M_PIf64 * u * (2*x + 1) / 32);
                    sum += val;
                }
            }
            sum = sum / 8.0;
            IDCT_IMG.at<double>(cv::Point(begin_x + x, begin_y + y)) = sum;
        }
    }
}

void FltToChar(cv::Mat &S_C_IMG)
{
    double min, max;
    cv::minMaxIdx(IDCT_IMG, &min, &max);
    cout << min << max << endl;
    if(min < 0) min = 0;
    if(max > 255) max = 255;
    cv::normalize(IDCT_IMG, S_C_IMG, (int)min, (int)max, cv::NORM_MINMAX, CV_8UC1);
}

int main(int argc, char **argv)
{
    if(argc < 1) {
        cout << "Please input file name" << endl;
        return 0;
    }
    if(!LoadImage(argv[1])) {
        cout << "Load Image error!" << endl;
        return 0;
    }
    cout << ORIGIN_IMG.channels() << endl;

    /*
    cv::namedWindow("origin image", cv::WINDOW_AUTOSIZE);
    cv::imshow("origin image", ORIGIN_IMG);
    cv::waitKey(0);
    */

    cv::split(ORIGIN_IMG, BGR_SPLIT_IMG);

    for(int i = 0; i < 3; ++i) {
        IDCT_IMG = cv::Mat::zeros(BGR_SPLIT_IMG[i].size(), CV_64FC1);
        for(int a = 0; a < BGR_SPLIT_IMG[i].cols / 16; ++a) { // x
            for(int b = 0; b < BGR_SPLIT_IMG[i].rows / 16; ++b) { // y
                cv::Mat coeff = cv::Mat::zeros(cv::Size(16, 16), CV_64FC1);
                MY_DCT(a * 16, b * 16, coeff, BGR_SPLIT_IMG[i]); 
                Select16Coeff(coeff);
                MY_IDCT(a * 16, b * 16, coeff);
            }
        }
        FltToChar(BGR_SPLIT_IMG[i]);
        IDCT_IMG.~Mat();
    }

    cv::merge(BGR_SPLIT_IMG, 3, MODIFY_IMG);

    string savePath = "DCT_" + string(argv[1]);
    cv::imwrite(savePath, MODIFY_IMG);

    //cv::resize(MODIFY_IMG, MODIFY_IMG, cv::Size(0, 0), 4.0, 4.0);

    cv::namedWindow("modify image", cv::WINDOW_AUTOSIZE);
    cv::imshow("modify image", MODIFY_IMG);
    cv::waitKey(0);

    return 0;
}