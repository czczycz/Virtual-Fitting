#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

Mat tranformMatrix(vector<String>& fileName, Point upLeft, Point lowRight, Mat prehomo, int flag)
{
clock_t startTime, endTime;

Mat img1=imread(fileName[0], IMREAD_GRAYSCALE);
Mat img2=imread(fileName[1], IMREAD_GRAYSCALE);

//    if(img1.rows * img1.cols <=0)
//    {
//        cout << "no img1" << endl;
//        return;
//    }
//    else{
//        cout << img1.cols << ", " << img1.rows << endl;
//    }
//    if(img2.rows * img2.cols <=0)
//    {
//        cout << "no img1" << endl;
//        return;
//    }
//    else{
//        cout << img2.cols << ", " << img2.rows << endl;
//    }


Ptr<Feature2D> b;

Ptr<DescriptorMatcher> descriptorMatcher;
vector<DMatch> matches;
vector<KeyPoint> keyImg1, keyImg2;
Mat descImg1, descImg2;

b = ORB::create(500);

startTime=clock();
b->detectAndCompute(img1, Mat(), keyImg1, descImg1, false);
endTime = clock();
cout << "first image feature detect time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

startTime=clock();
b->detectAndCompute(img2, Mat(),keyImg2, descImg2, false);
endTime = clock();
cout << "second image feature detect time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

startTime=clock();
descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
descriptorMatcher->match(descImg1, descImg2, matches, Mat());
endTime = clock();
cout << "feature matching time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

startTime=clock();
Mat index;
int nbMatch=int(matches.size());
Mat tab(nbMatch, 1, CV_32F);
for (int i = 0; i<nbMatch; i++)
{
tab.at<float>(i, 0) = matches[i].distance;
}
sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
vector<DMatch> bestMatches;
vector<Point2f> imagePoints1, imagePoints2;

for (int i = 0; i<nbMatch; i++)
{

KeyPoint kp=keyImg1[index.at<int>(i, 0)];
if(kp.pt.x < upLeft.x || kp.pt.x > lowRight.x || kp.pt.y < upLeft.y || kp.pt.y >lowRight.y) continue;

DMatch current=matches[index.at<int>(i, 0)];
bestMatches.push_back(current);

imagePoints1.push_back(keyImg1[current.queryIdx].pt);
imagePoints2.push_back(keyImg2[current.trainIdx].pt);

if(bestMatches.size()>=30) break;
//if(imagePoints1.size()>=30) break;
}

Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);
cout<<"Transform Matrix：\n"<<homo<<endl<<endl;
endTime = clock();
cout << "homo matrix calculate time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

Mat result;
rectangle(img1, upLeft, lowRight, cvScalar(1), 1);
drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
imshow("result", result);
imwrite("result.jpg",result);

vector<Point2d> rectPts(4), transPts(4);
rectPts[0]=upLeft;
rectPts[1]=Point(upLeft.x, lowRight.y);
rectPts[2]=lowRight;
rectPts[3]=Point(lowRight.x, upLeft.y);

Point points[1][4];

if(flag)
{
homo=prehomo*homo;
}

perspectiveTransform(rectPts, transPts, homo);
for(int i=0; i<4; i++)
{
//cout << transPts[i] << endl;
points[0][i]=transPts[i];
}

const Point* pt[1] = { points[0] };
int npt[1] = {4};
//polylines(img2, pt, npt, 1, 1, Scalar(250,0,0)) ;
polylines(img2, pt, npt, 1, 1, Scalar(1),1);
//imshow("img2_poly", img2);

Mat transResult(img1.rows,img1.cols + img2.cols+1,img1.type());
img1.colRange(0,img1.cols).copyTo(transResult.colRange(0,img1.cols));
img2.colRange(0,img2.cols).copyTo(transResult.colRange(img2.cols+1,transResult.cols));
imshow("tansResult",transResult);
imwrite("tansResult.jpg",transResult);

waitKey();
return homo;
}

void calTime()
{
vector<String> fileName, fileName1;
//fileName.push_back("image/001.jpg");
//fileName.push_back("image/002.jpg");
fileName.push_back("image/initialSmall01.jpg");

//fileName.push_back("image/shoes03Small01.jpg");
fileName.push_back("image/shoes03Small01.jpg");
//    fileName.push_back("image/90.jpg");
//    fileName.push_back("image/105.jpg");


clock_t startTime, endTime;
startTime = clock();
//featureMatch(fileName);
Mat homo1 = tranformMatrix(fileName, Point(215, 300), Point(310, 450), Mat(), 0);


fileName1.push_back("image/shoes03Small01.jpg");
fileName1.push_back("image/shoes04Small01.jpg");
Mat homo2 = tranformMatrix(fileName1, Point(200, 300), Point(295, 450), homo1, 1);
//    tranformMatrix(fileName, Point(430, 600), Point(620, 900));
//tranformMatrix(fileName, cvPoint(450, 620), Point(600, 780));
//tranformMatrix(fileName, Point(0, 0), Point(2247, 3263));
//tranformMatrix(fileName, Point(348,335), Point(552,786));
endTime = clock();
cout << "time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

return;

}

int main(int argc, char *argv[])
{
calTime();

return 0;
}














