#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <filesystem> // using directory listing
#include <algorithm>
#include <vector>
//#include <utility>

using namespace std;
namespace fs = std::filesystem; // need -std=c++17

string matAFilePath = "/home/ccma750/Documents/Face_dataset/matA.xml";
string dataDirPath = "/home/ccma750/Documents/Face_dataset/";
/*
string matWFilePath = "/home/ccma750/Documents/Face_dataset/matW.xml";
string matUFilePath = "/home/ccma750/Documents/Face_dataset/matU.xml";
string matVTFilePath = "/home/ccma750/Documents/Face_dataset/matVT.xml";
*/
//cv::Mat matA, matU, matW, matVT;
cv::Mat matA;
cv::PCA pca;
cv::Mat inputData; //input Data
int maxComponents = 1024;
double priority[1024] = {0, };

bool Lcompare(const pair<int, double>& a, const pair<int, double>& b)
{
	return a.second < b.second;
}

bool MakeDatasetByYale()
{
    string yaleDirPath = "/home/ccma750/Documents/Face_dataset/CroppedYale/";
    string imgSaveDirPath = "/home/ccma750/Documents/Face_dataset/TrainingDataSet/1_yale/";

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

bool MakeDatasetByCelebA(int argCnt = 0)
{
    string CelebADirPath = "/home/ccma750/Documents/Face_dataset/CelebA/img_align_celeba/";
    string imgSaveDirPath = "/home/ccma750/Documents/Face_dataset/TrainingDataSet/2_CelebA/";

    // open CelebA landmarks list txt
    ifstream landmarkListTxt("/home/ccma750/Documents/Face_dataset/CelebA/list_landmarks_align_celeba.txt", ifstream::in);
    string attribute;
    int minX, maxX, minY, maxY, width, height;
    int rect1, rect2, rect3, rect4;
    int datasetCnt = 0;

    if(argCnt < 0) {
        cout << "argCnt must be equal or greater than 0" << endl;
        return false;
    }

    getline(landmarkListTxt, attribute); // 202599
    getline(landmarkListTxt, attribute); // lefteye_x(10~12) lefteye_y(13~17) righteye_x(18~22) righteye_y(23~27)
                                         // nose_x(28~32) nose_y(33~37) leftmouth_x(38~42) leftmouth_y(43~47)
                                         // rightmouth_x(48~52) rightmouth_y(53~57)

    for(const auto & entry : fs::directory_iterator(CelebADirPath)) {
        string imgFilePath = entry.path();
        string imgFileName = entry.path().filename();
        string imgSavePath = imgSaveDirPath + imgFileName.substr(0, imgFileName.length()-4) + ".pgm";

        if(argCnt == datasetCnt) {
            cout << "datasetCnt : " << datasetCnt << " Finished!" << endl;
            break;
        }
        datasetCnt++;

        if(fs::exists(imgSavePath)) {
            cout << "Skip : " << imgSavePath << endl;
            continue;
        }

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

/*
 * arg Cnt = 0 means read full count
 */
bool MakeMatA(int argCnt = 0)
{
    string DatasetPath = "/home/ccma750/Documents/Face_dataset/TrainingDataSet/";
    int datasetCnt = 0;
    int datasetIdx = 0;

    if(argCnt < 0) {
        cout << "argCnt must be equal or greater than 0" << endl;
        return false;
    }

    for(const auto & entry : fs::recursive_directory_iterator(DatasetPath)) {
        if(entry.path().extension().generic_string().compare(".pgm") == 0) { // find pgm image.
            datasetCnt++;
        }
        if(argCnt == datasetCnt) {
            break;
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
        if(datasetIdx == datasetCnt) {
            break;
        }
    }

    return true;
}

void SaveMatA()
{
    // Export MatA to File
    //cv::FileStorage matAFile(matAFilePath, cv::FileStorage::WRITE);
    string filePath = dataDirPath + "matA" + to_string(matA.cols) + ".xml";
    cv::FileStorage matAFile(filePath, cv::FileStorage::WRITE);
    matAFile << "matA" << matA;
    matAFile.release();
}

bool LoadMatA(int argCnt = 3000)
{
    //cv::FileStorage matAFile(matAFilePath, cv::FileStorage::READ);
    string filePath = dataDirPath + "matA" + to_string(argCnt) + ".xml";
    cv::FileStorage matAFile(filePath, cv::FileStorage::READ);
    if(!matAFile.isOpened()) {
        cout << "Open MatA failed!" << endl;
        return false;
    }
    matAFile["matA"] >> matA;
    matAFile.release();

    return true;
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
    pca = cv::PCA(matA, cv::noArray(), cv::PCA::DATA_AS_COL, maxComponents);
}

void SavePCA()
{
    //cv::FileStorage fs("/home/ccma750/Documents/Face_dataset/pca.xml", cv::FileStorage::WRITE);
    string filePath = dataDirPath + "pca" + to_string(matA.cols) + ".xml";
    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    pca.write(fs);
}

bool LoadPCA(int argCnt = 3000)
{
    //cv::FileStorage fs("/home/ccma750/Documents/Face_dataset/pca.xml", cv::FileStorage::READ);
    string filePath = dataDirPath + "pca" + to_string(argCnt) + ".xml";
    cv::FileStorage fs(filePath, cv::FileStorage::READ);
    if(!fs.isOpened()) {
        cout << "Open PCA failed!" << endl;
        return false;
    }
    pca.read(fs.root());
    return true;
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
            if(maxComponents <= i*width + j) {
                i = height;
                break;
            }
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
    int matchedCnt = 0;
    cv::Mat coefficients = pca.project(inputData.t()); // [50 * maxComponents(1024)]
    //cout << coefficients.size() << endl;
    cv::Mat pri = cv::Mat(maxComponents, 1, CV_64FC1, priority);

    cout << coefficients.col(1).size() << endl;
    cout << pri.size() << endl;

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
            /*
            double dist = cv::norm(coefficients.col(i).mul(pca.eigenvalues), 
                                   coefficients.col(j*5).mul(pca.eigenvalues), cv::NORM_L2);
            */
            //double dist = cv::norm(coefficients.col(i)/pca.eigenvalues, coefficients.col(j*5)/pca.eigenvalues, cv::NORM_L2);
            //double dist = cv::norm(coefficients.col(i), coefficients.col(j*5), cv::NORM_L2);
            double dist = cv::norm(coefficients.col(i).mul(pri), 
                                   coefficients.col(j*5).mul(pri), cv::NORM_L2);
            if(dist < minDist) {
                preFace = j;
                minDist = dist;
            }
        }
        /*
        cout << "i : " << i << "pre : " << preFace << endl;
        */
        if(i/5 == preFace) {
            matchedCnt++;
        }
    }

    cout << "predict Rate: " << (double)matchedCnt / 40 << endl;
}

void PriOfEigenvectors()
{
    vector<pair<int, double>> stdDev;
    // Calc mean of variance of same human's coefficients and ordering Eigenvectors.
    cv::Mat coefficients = pca.project(inputData.t()); // [maxComponents(1024) * 50]
    cv::Mat tmpVar;

    for(int j = 0; j < 50; j += 5)
    {
        stdDev.clear();
        for(int i = 0; i < maxComponents; ++i)
        {
            cv::meanStdDev(coefficients(cv::Rect(j, i, 5, 1)).t(), cv::noArray(), tmpVar);
            stdDev.push_back(pair(i, tmpVar.at<double>(0)));
        }
        sort(stdDev.begin(), stdDev.end(), Lcompare); // Ascending order

        // Debug print
        /*
        for(int i = 0; i < maxComponents; ++i)
        {
            cout << stdDev[i].first << " : " << stdDev[i].second << endl;
        }
        */
        for(int i = 0; i < maxComponents; ++i)
        {
            priority[stdDev[i].first] += (double)(maxComponents-i);
        }
    }
    for(int i = 0; i < maxComponents; ++i)
    {
        cout << "i : " << priority[i] << endl;
    }

    //cv::meanStdDev(coefficients.rowRange(0, 5), cv::noArray(), tmpVar);
}

int main()
{
    //maxComponents = 50;
    LoadMatA(10000);
    LoadPCA(10000);
    //RunPCA();
    LoadInputData();
    PriOfEigenvectors();
    Predict();


    return 0;
}