#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <filesystem> // using directory listing
#include <random>

#define CompN 64
#define SAMPLE_CNT 20

using namespace std;
namespace fs = std::filesystem; // need -std=c++17

void DoDFT(string, cv::Mat&, double, bool, bool, double, double);
void PrintMag(const cv::Mat&);
void DoIDFT(const cv::Mat&);
void DFTofFabrics(string);

cv::Mat sample_vec[SAMPLE_CNT];
cv::Mat dft_mags[SAMPLE_CNT];

void DoDFT(string input_path, cv::Mat &magMat, double scale_factor, bool printDFT, bool printIDFT, double x_ratio = 0.0, double y_ratio = 0.0)
{
    cv::Mat inputImg = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    cv::Mat dftImg;
    cv::Mat dftImgArr[2];

    int rect_size;
    if(inputImg.cols < inputImg.rows)
        rect_size = inputImg.cols;
    else
        rect_size = inputImg.rows;
    inputImg = inputImg(cv::Rect(0, 0, rect_size, rect_size));

    // CV_8UC1 to CV_32FC1
    int x = rect_size * x_ratio;
    int y = rect_size * y_ratio;
    int crop_size = rect_size * scale_factor;

    inputImg = inputImg(cv::Rect(x, y, crop_size, crop_size));
    cv::resize(inputImg, inputImg, cv::Size(CompN, CompN));
    inputImg.convertTo(inputImg, CV_32FC1);

    // Do DFT
    cv::dft(inputImg,dftImg, cv::DFT_COMPLEX_OUTPUT);

    // Split real part and imaginary part
    cv::split(dftImg, dftImgArr);

    // Calc DFT magnitude
    cv::magnitude(dftImgArr[0], dftImgArr[1], magMat);

    // It is optional.
    // Print Magnitude.
    if(printDFT)
        PrintMag(magMat);
    // It is optional.
    // Print IDFT image.
    if(printIDFT)
        DoIDFT(dftImg);
    
    // Extract half
    //cv::log(magMat, magMat);
    //cv::resize(magMat, magMat, cv::Size(CompN, CompN));
    //magMat = magMat.row(CompN/2);

    // Extract Triangle matrix. where x + y < CompN
    /*
    cv::Mat opMagMat = cv::Mat(1, 0, CV_32FC1); //optimize magnitude matrix
    for(int y = 0; y < CompN; ++y) {
        cv::hconcat(opMagMat, magMat(cv::Rect(0, y, CompN-y, 1)), opMagMat);
    }
    opMagMat.copyTo(magMat);
    */
}

void PrintMag(const cv::Mat &magMat)
{
    cv::Mat dftPrint;
    magMat.copyTo(dftPrint);

    cv::log(dftPrint, dftPrint);

    // rearrange
    int halfX = dftPrint.cols/2;
    int halfY = dftPrint.rows/2;
    cv::Mat tmp;
    // swap 1st quadrant and 3rd quadrant
    dftPrint(cv::Rect(0, 0, halfX, halfY)).copyTo(tmp);
    dftPrint(cv::Rect(halfX, halfY, halfX, halfY)).copyTo(dftPrint(cv::Rect(0, 0, halfX, halfY)));
    tmp.copyTo(dftPrint(cv::Rect(halfX, halfY, halfX, halfY)));
    // swap 2nd quadrant and 4th quadrant
    dftPrint(cv::Rect(0, halfY, halfX, halfY)).copyTo(tmp);
    dftPrint(cv::Rect(halfX, 0, halfX, halfY)).copyTo(dftPrint(cv::Rect(0, halfY, halfX, halfY)));
    tmp.copyTo(dftPrint(cv::Rect(halfX, 0, halfX, halfY)));

    // Print
    cv::normalize(dftPrint, dftPrint, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::namedWindow("2D DFT image", cv::WINDOW_AUTOSIZE);
    cv::imshow("2D DFT image", dftPrint);
    cv::waitKey(0);
}

void DoIDFT(const cv::Mat &dftMat)
{
    cv::Mat idftMat;
    cv::dft(dftMat, idftMat, cv::DFT_SCALE|cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);

    // It is optional. Print idft image
    idftMat.convertTo(idftMat, CV_8UC1);
    cv::namedWindow("IDFT image", cv::WINDOW_AUTOSIZE);
    cv::imshow("IDFT image", idftMat);
    cv::waitKey(0);
}

void SaveMagAsImg(const cv::Mat &saveImg, int idx)
{
    cv::Mat dftPrint;
    saveImg.copyTo(dftPrint);

    cv::log(dftPrint, dftPrint);

    // rearrange
    int halfX = dftPrint.cols/2;
    int halfY = dftPrint.rows/2;
    cv::Mat tmp;
    // swap 1st quadrant and 3rd quadrant
    dftPrint(cv::Rect(0, 0, halfX, halfY)).copyTo(tmp);
    dftPrint(cv::Rect(halfX, halfY, halfX, halfY)).copyTo(dftPrint(cv::Rect(0, 0, halfX, halfY)));
    tmp.copyTo(dftPrint(cv::Rect(halfX, halfY, halfX, halfY)));
    // swap 2nd quadrant and 4th quadrant
    dftPrint(cv::Rect(0, halfY, halfX, halfY)).copyTo(tmp);
    dftPrint(cv::Rect(halfX, 0, halfX, halfY)).copyTo(dftPrint(cv::Rect(0, halfY, halfX, halfY)));
    tmp.copyTo(dftPrint(cv::Rect(halfX, 0, halfX, halfY)));

    // Print
    cv::normalize(dftPrint, dftPrint, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    string path = "/home/ccma750/VM_Shared_Directory/수치해석_HW10_Images/DFT_Real_Fabrics/" + to_string(idx) + ".jpg";
    cv::imwrite(path, dftPrint);
}

void DFTofFabrics(string dir_path)
{
    cv::Mat magMat;
    cv::Mat magMat2;
    cv::Mat optimal_vec;
    double op_dist;
    double dist;
    int idx;
    int sample_idx = 0;
    for(const auto & entry : fs::directory_iterator(dir_path)) {
        op_dist = DBL_MAX;
        idx = 0;
        cv::Mat avgVec = cv::Mat::zeros(1, 2*CompN*CompN/4/4, CV_32FC1);
        cv::Mat retVec[4];
        for(int i = 25; i <= 50; i += 5) {
            double scale = (double)i/100;
            DoDFT(entry.path().generic_string(), retVec[0], scale, false, false, 0.0, 0.0);
            DoDFT(entry.path().generic_string(), retVec[1], scale, false, false, scale, 0.0);
            DoDFT(entry.path().generic_string(), retVec[2], scale, false, false, 0.0, scale);
            DoDFT(entry.path().generic_string(), retVec[3], scale, false, false, scale, scale);

            for(int j = 0; j < 4; ++j) {
                retVec[j] = retVec[j].rowRange(0, CompN/2);
                // 1/8 of DFTs...
                cv::hconcat(retVec[j](cv::Rect(0, 0, CompN/4, CompN/4)), retVec[j](cv::Rect(3*CompN/4, 0, CompN/4, CompN/4)), retVec[j]);
                /* Triangle of DFTs...
                for(int k = 0; k < CompN/2; ++k) {
                    for(int l = 0; l < k; ++l) {
                        retVec[j].at<float>(k, CompN/2-l) = 0.0;
                        retVec[j].at<float>(k, CompN/2+l) = 0.0;
                    }
                }
                */
                retVec[j] = retVec[j].reshape(0, 1);
                //retVec[j] = retVec[j].colRange(1, retVec[j].cols); // Remove DC
                retVec[j].at<float>(0, 0) = 0.0;
                avgVec = avgVec + retVec[j];
            }
            avgVec = avgVec / 4;
            dist = cv::norm(retVec[0], avgVec, cv::NORM_L2);
            if(dist < op_dist) {
                avgVec.copyTo(sample_vec[sample_idx]);
                op_dist = dist;
                idx = i;
            }
        }
        /*
        avgVec = cv::Mat::zeros(CompN, CompN, CV_32FC1);
        double scale = (double)idx/100;
        DoDFT(entry.path().generic_string(), retVec[0], scale, false, false, 0.0, 0.0);
        DoDFT(entry.path().generic_string(), retVec[1], scale, false, false, scale, 0.0);
        DoDFT(entry.path().generic_string(), retVec[2], scale, false, false, 0.0, scale);
        DoDFT(entry.path().generic_string(), retVec[3], scale, false, false, scale, scale);
        for(int j = 0; j < 4; ++j) {
            avgVec = avgVec + retVec[j];
        }
        avgVec.copyTo(dft_mags[sample_idx]);
        SaveMagAsImg(dft_mags[sample_idx], sample_idx);
        */

        /*
        op_dist = DBL_MAX;
        idx = 0;
        for(int i = 10; i <= 50; i += 5) {
            double scale = (double)i/100;
            DoDFT(entry.path().generic_string(), magMat, scale, false, false);
            DoDFT(entry.path().generic_string(), magMat2, scale, false, false, scale, scale);

            magMat = magMat.reshape(0, 1);
            magMat2 = magMat2.reshape(0, 1);
            magMat = magMat.colRange(1, magMat.cols); // Remove DC
            magMat2 = magMat2.colRange(1, magMat2.cols); // Remove DC
            //cv::log(magMat, magMat); // log
            //cv::log(magMat2, magMat2); // log

            dist = cv::norm(magMat, magMat2, cv::NORM_L2);
            if(dist < op_dist) {
                magMat.copyTo(sample_vec[sample_idx]);
                op_dist = dist;
                idx = i;
            }
        }
        */
        

        /*
        DoDFT(entry.path().generic_string(), magMat, 1.0, false, false);
        //cv::resize(magMat, optimal_vec, cv::Size(CompN, CompN)); // Do in DoDFT.

        magMat = magMat.reshape(0, 1);
        magMat.colRange(1, magMat.cols).copyTo(optimal_vec); // Remove DC
        optimal_vec.copyTo(sample_vec[sample_idx]);
        */

        /*
        // Using vector distance
        op_dist = DBL_MAX;
        idx = 0;
        for(int i = 1; i < 10; ++i) {
            DoDFT(entry.path().generic_string(), magMat, (double)i/10, false, false);

            magMat = magMat.reshape(0, 1);
            magMat = magMat.colRange(1, magMat.cols); // Remove DC

            dist = cv::norm(optimal_vec, magMat, cv::NORM_L2);
            //cout << "idx : " << i << "\tdist : " << dist << endl;
            if(dist < op_dist) {
                magMat.copyTo(sample_vec[sample_idx]);
                op_dist = dist;
                idx = i;
            }
        }
        */
        
        /*
        //Using cosine similarity
        op_dist = DBL_MIN;
        idx = 0;
        for(int i = 1; i < 10; ++i) {
            DoDFT(entry.path().generic_string(), magMat, (double)i/10, false, false);

            magMat = magMat.reshape(0, 1);
            magMat = magMat.colRange(1, magMat.cols); // Remove DC
            //cv::log(magMat, magMat); // log

            //dist = cv::norm(optimal_vec, magMat, cv::NORM_L2);
            dist = optimal_vec.dot(magMat) / (cv::norm(optimal_vec) * cv::norm(magMat));
            cout << "idx : " << i << "\tdist : " << dist << endl;
            if(dist > op_dist) {
                magMat.copyTo(sample_vec[sample_idx]);
                op_dist = dist;
                idx = i;
            }
        }
        */
        sample_idx++;

        cout << "filename: " << entry.path().filename() << endl;
        cout << "sample idx: " << sample_idx << "\tidx : " << idx << "\top_dist : " << op_dist << endl;

        // Print Image
        /*
        cv::Mat tmp_img = cv::imread(entry.path());
        tmp_img = tmp_img(cv::Rect(0, 0, tmp_img.cols*idx/10, tmp_img.rows*idx/10));
        cv::namedWindow("Cropped image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Cropped image", tmp_img);
        cv::waitKey(0);
        */
    }
}

void PredictPatterns(string dir_path)
{
    cv::Mat magMat;
    cv::Mat optimal_vec;
    double op_dist;
    double dist;
    int sample_idx = 0;
    int predict_idx = 0;
    int correct_cnt = 0;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> re_gen(0.25, 0.5);

    for(const auto & entry : fs::directory_iterator(dir_path)) {
        for(int i = 0; i < 5; ++i) {
            double x_ratio, y_ratio, scale_factor;
            x_ratio = re_gen(gen);
            y_ratio = re_gen(gen);
            scale_factor = re_gen(gen);
            DoDFT(entry.path().generic_string(), magMat, scale_factor, false, false, x_ratio, y_ratio);

            magMat = magMat.rowRange(0, CompN/2);
            // 1/8 of DFTs...
            cv::hconcat(magMat(cv::Rect(0, 0, CompN/4, CompN/4)), magMat(cv::Rect(3*CompN/4, 0, CompN/4, CompN/4)), magMat);
            /* Triangle of DFTs...
            for(int k = 0; k < CompN/2; ++k) {
                    for(int l = 0; l < k; ++l) {
                        magMat.at<float>(k, CompN/2-l) = 0.0;
                        magMat.at<float>(k, CompN/2+l) = 0.0;
                    }
            }
            */
            magMat = magMat.reshape(0, 1);
            
            //magMat = magMat.colRange(1, magMat.cols); // Remove DC
            magMat.at<float>(0, 0) = 0.0;
            //cv::log(magMat, magMat); // log

            predict_idx = -1;
            op_dist = DBL_MAX;
            for(int j = 0; j < SAMPLE_CNT; ++j) {
                dist = cv::norm(sample_vec[j], magMat, cv::NORM_L2);
                if(dist < op_dist) {
                    op_dist = dist;
                    predict_idx = j;
                }
            }

            cout << "sample_idx: " << sample_idx << "\tpredict_idx: " << predict_idx << endl;
            if(predict_idx == sample_idx) {
                correct_cnt++;
            }
        }
        sample_idx++;
    }
    cout << "predict rate: " << (double)correct_cnt/100 << endl;
}

int main(int argc, char **argv)
{
    string input_path;
    if(argc < 2) {
        cout << "Please input directory path!" << endl;
        return 0;
    }
    input_path = argv[1];

    //input_path = "/home/ccma750/VM_Shared_Directory/수치해석_HW10_Images/easypattern/";
    input_path = "/home/ccma750/VM_Shared_Directory/수치해석_HW10_Images/Real_Fabrics/";

    cout << "Start DFT of Sample Patterns" << endl;
    DFTofFabrics(input_path);


    cout << "Start Predict!" << endl;
    PredictPatterns(input_path);
    

    return 0;
}