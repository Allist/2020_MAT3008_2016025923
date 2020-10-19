#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <filesystem> // using directory listing

using namespace std;
namespace fs = std::filesystem; // need -std=c++17

string matAFilePath = "/home/ccma750/Documents/Face_dataset/matA.xml";
/*
string matWFilePath = "/home/ccma750/Documents/Face_dataset/matW.xml";
string matUFilePath = "/home/ccma750/Documents/Face_dataset/matU.xml";
string matVTFilePath = "/home/ccma750/Documents/Face_dataset/matVT.xml";
*/
//cv::Mat matA, matU, matW, matVT;
cv::Mat matA;
cv::PCA pca;
cv::Mat inputData; //input Data

bool MakeDatasetByYale()
{
    string yaleDirPath = "/home/ccma750/Documents/Face_dataset/CroppedYale/";
    string imgSaveDirPath = "/home/ccma750/Documents/Face_dataset/TrainingDataSet/yale/";

    for(const auto & entry : fs::directory_iterator(yaleDirPath)) {
        string imgFileName = entry.path().filename();
        string imgFilePath = entry.path().generic_string() + "/" + imgFileName + "_P00A+000E+00.pgm";
        string imgSavePath = imgSaveDirPath + imgFileName + ".pgm";

        // Open image and resize 32x32 and save it.
        cv::Mat origin_img = cv::imread(imgFilePath, cv::IMREAD_GRAYSCALE);
        cv::Mat compressed_image; // this image is compressed image
        cv::resize(origin_img, compressed_image, cv::Size(32, 32), 0, 0);

        if(compressed_image.empty()){
            cout << "Error opening image : " << imgFilePath << endl;
            return false;
        }

        if(cv::imwrite(imgSavePath, compressed_image)) {
            cout << "Success: " << imgSavePath << endl;
        } else {
            cout << "Failed: " << imgSavePath << endl;
            return false;
        }
    }

    return true;
}

bool MakeDatasetByCelebA()
{
    string CelebADirPath = "/home/ccma750/Documents/Face_dataset/CelebA/img_align_celeba/";
    string imgSaveDirPath = "/home/ccma750/Documents/Face_dataset/TrainingDataSet/CelebA/";

    // open CelebA landmarks list txt
    ifstream landmarkListTxt("/home/ccma750/Documents/Face_dataset/CelebA/list_landmarks_align_celeba.txt", ifstream::in);
    string attribute;
    int minX, maxX, minY, maxY, width, height;
    int rect1, rect2, rect3, rect4;

    getline(landmarkListTxt, attribute); // 202599
    getline(landmarkListTxt, attribute); // lefteye_x(10~12) lefteye_y(13~17) righteye_x(18~22) righteye_y(23~27)
                                         // nose_x(28~32) nose_y(33~37) leftmouth_x(38~42) leftmouth_y(43~47)
                                         // rightmouth_x(48~52) rightmouth_y(53~57)

    for(const auto & entry : fs::directory_iterator(CelebADirPath)) {
        string imgFilePath = entry.path();
        string imgFileName = entry.path().filename();
        string imgSavePath = imgSaveDirPath + imgFileName.substr(0, imgFileName.length()-4) + ".pgm";

        {// Find min/max of x/y
            getline(landmarkListTxt, attribute);
            maxX = maxY = 0;
            minX = minY = INT32_MAX;
            int x = stoi(attribute.substr(10, 3));
            int y = stoi(attribute.substr(13, 5));
            if(x < minX) minX = x;
            else if(maxX < x) maxX = x;
            if(y < minY) minY = y;
            else if(maxY < y) maxY = y;

            for(int i = 0; i < 4; ++i) {
                x = stoi(attribute.substr(18 + 10*i, 5));
                y = stoi(attribute.substr(23 + 10*i, 5));
                if(x < minX) minX = x;
                else if(maxX < x) maxX = x;
                if(y < minY) minY = y;
                else if(maxY < y) maxY = y;
            }
        }

        {// Calc width, height, add padding
            //cout << minX << maxX << minY << maxY << endl;
            width = maxX - minX;
            height = maxY - minY;

            rect1 = minX - width*0.25;
            rect2 = minY - height*0.25;
            rect3 = width*1.5;
            rect4 = height*1.5;
            //cout << rect1 << rect2 << rect3 << rect4 << endl;
        }

        cv::Mat origin_img = cv::imread(imgFilePath, cv::IMREAD_GRAYSCALE);
        cv::Mat cropped_image = origin_img(cv::Rect(rect1, rect2, rect3, rect4));
        cv::Mat compressed_image; // this image is compressed image
        cv::resize(cropped_image, compressed_image, cv::Size(32, 32), 0, 0);

        if(cv::imwrite(imgSavePath, compressed_image)) {
            cout << "Success: " << imgSavePath << endl;
        } else {
            cout << "Failed: " << imgSavePath << endl;
            return false;
        }



        /*
        // Open image and resize 32x32 and save it.
        //cv::Mat origin_img = cv::imread(imgFilePath, cv::IMREAD_GRAYSCALE);
        cv::Mat compressed_image; // this image is compressed image
        cv::resize(origin_img, compressed_image, cv::Size(32, 32), 0, 0);

        if(compressed_image.empty()){
            cout << "Error opening image : " << imgFilePath << endl;
            return false;
        }

        if(cv::imwrite(imgSavePath, compressed_image)) {
            cout << "Success: " << imgSavePath << endl;
        } else {
            cout << "Failed: " << imgSavePath << endl;
            return false;
        }
        */

        /*
        if(cv::imwrite(imgSavePath, compressed_image)) {
            cout << "Success: " << imgSavePath << endl;
        } else {
            cout << "Failed: " << imgSavePath << endl;
        }
        */

        /*
        cout << compressed_image.cols << endl;
        cout << compressed_image.rows << endl;

        cv::namedWindow("compressed_image", cv::WINDOW_AUTOSIZE);
        cv::imshow("compressed_image", compressed_image);
        cv::waitKey(0);
        */
    }

    return true;
}

bool MakePredictSetByYale()
{
    string yaleDirPath = "/home/ccma750/Documents/Face_dataset/CroppedYale/";
    string imgSaveDirPath = "/home/ccma750/Documents/Face_dataset/PredictDataSet/";

    vector<string> attributes;
    attributes.push_back("_P00A+000E+00.pgm");
    attributes.push_back("_P00A+025E+00.pgm");
    attributes.push_back("_P00A-025E+00.pgm");
    attributes.push_back("_P00A+000E-20.pgm");
    attributes.push_back("_P00A+000E-35.pgm");

    for(int i = 20; i < 30; ++i) { // 10 people
        for(vector<string>::iterator it = attributes.begin(); it != attributes.end(); ++it)
        {
            string imgFilePath = yaleDirPath + "yaleB" + to_string(i) + "/yaleB" + to_string(i) + *it;
            
            // Open image and resize 32x32 and save it.
            cv::Mat origin_img = cv::imread(imgFilePath, cv::IMREAD_GRAYSCALE);

            if(origin_img.data == NULL) {
                cout << "Error opening image : " << imgFilePath << endl;
                return false;
            }

            cv::Mat compressed_image; // this image is compressed image
            cv::resize(origin_img, compressed_image, cv::Size(32, 32), 0, 0);

            if(compressed_image.empty()){
                cout << "Error opening image : " << imgFilePath << endl;
                return false;
            }

            string imgSavePath = imgSaveDirPath + "yaleB" + to_string(i) + *it;
            if(cv::imwrite(imgSavePath, compressed_image)) {
                cout << "Success: " << imgSavePath << endl;
            } else {
                cout << "Failed: " << imgSavePath << endl;
                return false;
            }
        }
    }

    return true;
}

bool MakeMatA()
{
    string DatasetPath = "/home/ccma750/Documents/Face_dataset/TrainingDataSet/";
    int datasetCnt = 0;
    int datasetIdx = 0;

    for(const auto & entry : fs::recursive_directory_iterator(DatasetPath)) {
        if(entry.path().extension().generic_string().compare(".pgm") == 0) { // find pgm image.
            datasetCnt++;
        }
    }

    cout << "dataset Cnt: " << datasetCnt << endl;

    matA = cv::Mat(1024, datasetCnt, CV_64F);

    for(const auto & entry : fs::recursive_directory_iterator(DatasetPath)) {
        if(entry.path().extension().generic_string().compare(".pgm") != 0) { // Not pgm file.
            continue;
        }

        // Open image and copy image vector to matA
        cv::Mat data_img = cv::imread(entry.path().generic_string(), cv::IMREAD_GRAYSCALE);

        /*
        cout << matA.size() << endl;
        cout << data_img.size() << endl;
        cout << data_img.reshape(0, 1024).size() << endl;
        */

        //data_img.reshape(0, 1024).copyTo(matA(cv::Rect(datasetIdx, 0, 1, 1024)));
        data_img.reshape(0, 1024).convertTo(matA(cv::Rect(datasetIdx, 0, 1, 1024)), CV_64F);
        datasetIdx++;
    }

    return true;
}

void SaveMatA()
{
    // Export MatA to File
    cv::FileStorage matAFile(matAFilePath, cv::FileStorage::WRITE);
    matAFile << "matA" << matA;
    matAFile.release();
}

void LoadMatA()
{
    cv::FileStorage matAFile(matAFilePath, cv::FileStorage::READ);
    matAFile["matA"] >> matA;
    matAFile.release();
}

/*
void CalcSVD()
{
    cv::Mat w,u,vt;
    cv::SVD::compute(matA, w, u, vt, cv::SVD::FULL_UV);
    cv::FileStorage matWFile(matWFilePath, cv::FileStorage::WRITE);
    cv::FileStorage matUFile(matUFilePath, cv::FileStorage::WRITE);
    cv::FileStorage matVTFile(matVTFilePath, cv::FileStorage::WRITE);

    cout << matA.size() << endl;
    cout << w.size() << endl;
    cout << u.size() << endl;
    cout << vt.size() << endl;

    matWFile << "matW" << w;
    matUFile << "matU" << u;
    matVTFile << "matVT" << vt;
}
*/

/*
void LoadMat_W_U_VT()
{
    cv::FileStorage matWFile(matWFilePath, cv::FileStorage::READ);
    cv::FileStorage matUFile(matUFilePath, cv::FileStorage::READ);
    cv::FileStorage matVTFile(matVTFilePath, cv::FileStorage::READ);

    matWFile["matW"] >> matW;
    matUFile["matU"] >> matU;
    matVTFile["matVT"] >> matVT;
}
*/

void RunPCA()
{
    pca = cv::PCA(matA, cv::noArray(), cv::PCA::DATA_AS_COL, 0);
}

void SavePCA()
{
    cv::FileStorage fs("/home/ccma750/Documents/Face_dataset/pca.xml", cv::FileStorage::WRITE);
    pca.write(fs);
}

void LoadPCA()
{
    cv::FileStorage fs("/home/ccma750/Documents/Face_dataset/pca.xml", cv::FileStorage::READ);
    pca.read(fs.root());
}

void PrintEigenFaces()
{
    cv::Mat eigenvectors = pca.eigenvectors;
    cv::Mat eigenvector;
    cv::Mat eigenface;
    cv::Mat PrintMat = cv::Mat(cv::Size(32*10, 32*10), CV_8UC1);
    int width = 10;
    int height = 10;

    cout << eigenvectors.size() << endl;
    cout << pca.eigenvalues.size() << endl;

    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            eigenvector = eigenvectors(cv::Rect(0, i*width+j, 1024, 1));
            cv::normalize(eigenvector, eigenface, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            eigenface = eigenface.reshape(0, 32);
            eigenface.copyTo(PrintMat(cv::Rect(32*j, 32*i, 32, 32)));
        }
    }

    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image", PrintMat);
    cv::waitKey(0);
}

bool LoadInputData()
{
    string inputImgDirPath = "/home/ccma750/Documents/Face_dataset/PredictDataSet/";

    vector<string> attributes;
    attributes.push_back("_P00A+000E+00.pgm");
    attributes.push_back("_P00A+025E+00.pgm");
    attributes.push_back("_P00A-025E+00.pgm");
    attributes.push_back("_P00A+000E-20.pgm");
    attributes.push_back("_P00A+000E-35.pgm");

    for(int i = 20; i < 30; ++i) { // 10 people
        for(vector<string>::iterator it = attributes.begin(); it != attributes.end(); ++it)
        {
            string imgFilePath = inputImgDirPath + "yaleB" + to_string(i) + *it;
            
            // Open image
            cv::Mat input_img = cv::imread(imgFilePath, cv::IMREAD_GRAYSCALE);
            if(input_img.data == NULL) {
                cout << "Error opening image : " << imgFilePath << endl;
                return false;
            }

            inputData.push_back(input_img.reshape(0, 1));
        }
    }

    return true;
}

void Predict()
{
    cv::Mat coefficients = pca.project(inputData.t()); // [50 * 1024]
    cout << coefficients.size() << endl;

    for(int i = 0; i < 50; ++i)
    {
        if(i%5 == 0) // skip representative image
        {
            continue;
        }
        double minDist = DBL_MAX;
        int preFace = -1; // -1 is error.
        for(int j = 0; j < 10; ++j) // 10 people
        {
            double dist = cv::norm(coefficients.col(i), coefficients.col(j*5), cv::NORM_L2);
            if(dist < minDist) {
                preFace = j;
                minDist = dist;
            }
        }
        cout << "i : " << i << "pre : " << preFace << endl;
    }
}

int main()
{
    LoadMatA();
    LoadPCA();
    LoadInputData();
    Predict();

    return 0;
}