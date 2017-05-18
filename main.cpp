#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>

#include <iostream>

#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

void featureMatch(vector<String>& fileName)
{
    vector<String> typeDesc;
    vector<String> typeAlgoMatch;
    // This descriptor are going to be detect and compute
    //typeDesc.push_back("AKAZE");
    typeDesc.push_back("ORB");
    //typeDesc.push_back("BRISK");
    // This algorithm would be used to match descriptors
    //typeAlgoMatch.push_back("BruteForce");
    //typeAlgoMatch.push_back("BruteForce-L1");
    typeAlgoMatch.push_back("BruteForce-Hamming");
    typeAlgoMatch.push_back("BruteForce-Hamming(2)");

    //Mat img1 = imread("image/001.jpg", IMREAD_GRAYSCALE);
    //Mat img2 = imread("image/002.jpg", IMREAD_GRAYSCALE);
    Mat img1, img2;
    if(fileName.size()>=2)
    {
        img1 = imread(fileName[0], IMREAD_GRAYSCALE);
        img2 = imread(fileName[1], IMREAD_GRAYSCALE);
    }
    else
    {
        cout << "not enough image" << endl;
        return;
    }
    
    if (img1.rows*img1.cols <= 0)
    {
        cout << "Image " << fileName[0] << " is empty or cannot be found\n";
        return;
    }
    if (img2.rows*img2.cols <= 0)
    {
        cout << "Image " << fileName[1] << " is empty or cannot be found\n";
        return;
    }
    
    Ptr<Feature2D> b;
    Point upLeft = Point(348,335);
    Point lowRight = Point(552,786);
    //int FeaNum = 30;
    
    // Descriptor loop
    vector<String>::iterator itDesc;
    for (itDesc = typeDesc.begin(); itDesc != typeDesc.end(); ++itDesc)
    {
        Ptr<DescriptorMatcher> descriptorMatcher;
        // Match between img1 and img2
        vector<DMatch> matches;
        // keypoint  for img1 and img2
        vector<KeyPoint> keyImg1, keyImg2;
        // Descriptor for img1 and img2
        Mat descImg1, descImg2;
        vector<String>::iterator itMatcher = typeAlgoMatch.end();
        if (*itDesc == "AKAZE"){
            b = AKAZE::create();
        }
        if (*itDesc == "ORB"){
            //b = ORB::create(30, 1.5f, 2, 10, 0, 2, 0, 10);
            // b = ORB::create(500, 1.2f, 8, 10, 0, 2, 31, 20);
            b = ORB::create(100);
        }
        else if (*itDesc == "BRISK"){
            b = BRISK::create(50, 2, 1.0f);
        }
        try
        {
            clock_t startTime, endTime;
            
            startTime = clock();
            b->detectAndCompute(img1, Mat(), keyImg1, descImg1, false);
            endTime = clock();
            //cout << "keyImg1 : " << keyImg1.size() << endl;
            cout << "Image1 Feature detect time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
            
            startTime = clock();
            // or detect and compute descriptors in one step
            b->detectAndCompute(img2, Mat(),keyImg2, descImg2, false);
            endTime = clock();
            //cout << "keyImg2 : " << keyImg2.size() << endl;
            cout << "Image2 Feature detect time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
            
            // Match method loop
            for (itMatcher = typeAlgoMatch.begin(); itMatcher != typeAlgoMatch.end(); ++itMatcher){
                descriptorMatcher = DescriptorMatcher::create(*itMatcher);
                try
                {
                    startTime = clock();
                    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
                    // Keep best matches only to have a nice drawing.
                    // We sort distance between descriptor matches
                    Mat index;
                    int nbMatch=int(matches.size());
                    cout << "match size : " << nbMatch << endl;
                    Mat tab(nbMatch, 1, CV_32F);
                    for (int i = 0; i<nbMatch; i++)
                    {
                        tab.at<float>(i, 0) = matches[i].distance;
                    }
                    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
                    vector<DMatch> bestMatches;
                    vector<int> newIndex;
                    vector<int> originalIndex;
                    //inside Point(348,335) to Point(552,786)
                    for (int i = 0; i<nbMatch; i++)
                    {
                        KeyPoint kp=keyImg1[index.at<int>(i, 0)];
                        if(kp.pt.x < upLeft.x || kp.pt.x > lowRight.x || kp.pt.y < upLeft.y || kp.pt.y >lowRight.y) continue;
                        bestMatches.push_back(matches[index.at<int>(i, 0)]);
                        if(bestMatches.size()>=30) break;
                    }
                    endTime = clock();
                    cout << "Feature match time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
                    
                    Mat result;
                    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
                    namedWindow(*itDesc+": "+*itMatcher);
                    //resizeWindow(*itDesc+": "+*itMatcher, 100, 100);
                    imshow(*itDesc + ": " + *itMatcher, result);
                    //imwrite(*itDesc + ": " + *itMatcher+".jpg", result);
                    vector<DMatch>::iterator it;
                    cout << "Matches img1 img2" << endl;
                    for(it = bestMatches.begin(); it != bestMatches.end(); ++it)
                    {
                        cout << keyImg1[it->queryIdx].pt << "\t" << keyImg2[it->trainIdx].pt << endl;
                    }
                    waitKey();
                }
                catch (Exception& e)
                {
                    cout << e.msg << endl;
                    cout << "Cumulative distance cannot be computed." << endl;
                }
            }
        }
        catch (Exception& e)
        {
            cout << "Feature : " << *itDesc << "\n";
            if (itMatcher != typeAlgoMatch.end())
            {
                cout << "Matcher : " << *itMatcher << "\n";
            }
            cout << e.msg << endl;
        }
    }
    return;
}

void tranformMatrix(vector<String>& fileName, Point upLeft, Point lowRight)
{
    Mat img1=imread(fileName[0], IMREAD_GRAYSCALE);
    Mat img2=imread(fileName[1], IMREAD_GRAYSCALE);
    
    //cout << img1.cols << ", " << img1.rows << endl;
    //cout << img2.cols << ", " << img2.rows << endl;
    
    Ptr<Feature2D> b;
    //Point upLeft = Point(348,335);
    //Point lowRight = Point(552,786);
    
    Ptr<DescriptorMatcher> descriptorMatcher;
    vector<DMatch> matches;
    vector<KeyPoint> keyImg1, keyImg2;
    Mat descImg1, descImg2;
    
    b = ORB::create(500);
    b->detectAndCompute(img1, Mat(), keyImg1, descImg1, false);
    b->detectAndCompute(img2, Mat(),keyImg2, descImg2, false);
    
    descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
    
    Mat index;
    int nbMatch=int(matches.size());
    Mat tab(nbMatch, 1, CV_32F);
    for (int i = 0; i<nbMatch; i++)
    {
        tab.at<float>(i, 0) = matches[i].distance;
    }
    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
    vector<DMatch> bestMatches;
    vector<int> newIndex;
    vector<int> originalIndex;
    //inside Point(348,335) to Point(552,786)
    for (int i = 0; i<nbMatch; i++)
    {
        KeyPoint kp=keyImg1[index.at<int>(i, 0)];
        if(kp.pt.x < upLeft.x || kp.pt.x > lowRight.x || kp.pt.y < upLeft.y || kp.pt.y >lowRight.y) continue;
        bestMatches.push_back(matches[index.at<int>(i, 0)]);
        if(bestMatches.size()>=30) break;
    }
    
    Mat result;
    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
    namedWindow("result");
    imshow("result", result);
    imwrite("result.jpg",result);

    //获取排在前N个的最优匹配特征点
    vector<Point2f> imagePoints1,imagePoints2;
    
    for(int i=0;i<10;i++)
    {
        imagePoints1.push_back(keyImg1[bestMatches[i].queryIdx].pt);
        imagePoints2.push_back(keyImg2[bestMatches[i].trainIdx].pt);
    }
    
    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);
    cout<<"变换矩阵为：\n"<<homo<<endl<<endl; //输出映射矩阵

    //图像配准
    Mat imageTransform1;
    
    warpPerspective(img1,imageTransform1,homo,Size(img2.cols,img2.rows));
    imshow("Perspective matrix transformation",imageTransform1);

    
    Mat transResult(img1.rows,img1.cols + img2.cols+1,img1.type());
    imageTransform1.colRange(0,imageTransform1.cols).copyTo(transResult.colRange(0,imageTransform1.cols));
    img2.colRange(0,img2.cols).copyTo(transResult.colRange(img2.cols+1,transResult.cols));
    imshow("tansResult",transResult);
    imwrite("tansResult.jpg",transResult);
    
    waitKey();
    return;
}

void jointImage(vector<String>& fileName)
{
    Mat img1=imread(fileName[0]);
    Mat img2=imread(fileName[1]);
    Mat result(img1.rows,img1.cols + img2.cols+1,img1.type());
    
    img1.colRange(0,img1.cols).copyTo(result.colRange(0,img1.cols));
    img2.colRange(0,img2.cols).copyTo(result.colRange(img2.cols+1,result.cols));
    imshow("result",result);  
    waitKey(0);
    return;
}

void selectShoesArea()
{
    Mat initialImg=imread("image/initial.JPG");
    //Mat initialImg=imread("image/001.jpg");
    Mat dstImg;
    
    cout << "width is " << initialImg.cols << ", height is " << initialImg.rows << endl;
    resize(initialImg, dstImg, Size(1008, 1344), INTER_LINEAR);
    cout << "width is " << dstImg.cols << ", height is " << dstImg.rows << endl;
    
    rectangle(dstImg, cvPoint(430, 600), cvPoint(620, 900), cvScalar(1));
    
    imshow("dstImg", dstImg);
    waitKey();
    
    return;
}

int main(int argc, char *argv[])
{
    vector<String> fileName;
    //fileName.push_back("image/001.jpg");
    //fileName.push_back("image/002.jpg");
    fileName.push_back("image/75.JPG");
    fileName.push_back("image/60.JPG");
    
    //featureMatch(fileName);
    //tranformMatrix(fileName, Point(430, 600), cvPoint(620, 900));
    tranformMatrix(fileName, Point(0, 0), cvPoint(2247, 3263));

    //tranformMatrix(fileName, Point(348,335), Point(552,786));
    //jointImage(fileName);
    //selectShoesArea();
    return 0;
}













