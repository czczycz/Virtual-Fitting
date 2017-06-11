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


Mat tranformMatrix(vector<String>& fileName, int count, Mat previousMat, vector<Point2d>& transPoints, vector<Point2d>& initPoints)
{
    clock_t startTime, endTime;
    
    Mat img0=imread(fileName[0], IMREAD_GRAYSCALE);
    Mat img1=imread(fileName[count], IMREAD_GRAYSCALE);
    Mat img2=imread(fileName[count+1], IMREAD_GRAYSCALE);
    
    Ptr<Feature2D> b;
    
    Ptr<DescriptorMatcher> descriptorMatcher;
    vector<DMatch> matches;
    vector<KeyPoint> keyImg1, keyImg2;
    Mat descImg1, descImg2;
    
    b = ORB::create(500);
    
    startTime=clock();
    b->detectAndCompute(img1, Mat(), keyImg1, descImg1, false);
    endTime = clock();
    cout << "first image feature detect time : " <<(double)1000*(endTime - startTime) / CLOCKS_PER_SEC << "ms" << endl;
    
    startTime=clock();
    b->detectAndCompute(img2, Mat(),keyImg2, descImg2, false);
    endTime = clock();
    cout << "second image feature detect time : " <<(double)1000*(endTime - startTime) / CLOCKS_PER_SEC << "ms" << endl;
    
    startTime=clock();
    descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
    descriptorMatcher->match(descImg1, descImg2, matches, Mat());
    endTime = clock();
    cout << "feature matching time : " <<(double)1000*(endTime - startTime) / CLOCKS_PER_SEC << "ms" << endl;
    
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
        if((kp.pt.x < transPoints[0].x && kp.pt.x < transPoints[1].x) || (kp.pt.x > transPoints[2].x && kp.pt.x > transPoints[3].x) || (kp.pt.y < transPoints[0].y && kp.pt.y < transPoints[3].y) || (kp.pt.y >transPoints[2].y && kp.pt.y >transPoints[1].y)) continue;
        
        DMatch current=matches[index.at<int>(i, 0)];
        bestMatches.push_back(current);
 
        imagePoints1.push_back(keyImg1[current.queryIdx].pt);
        imagePoints2.push_back(keyImg2[current.trainIdx].pt);
        
        if(bestMatches.size()>=30) break;
        //if(imagePoints1.size()>=30) break;
    }
    
    Mat homo=findHomography(imagePoints1,imagePoints2,CV_RANSAC);
    if (count == 0) previousMat = homo;
    if (count > 0) {
        homo = homo*previousMat;
    }
    cout<<"Transform Matrix：\n"<<homo<<endl<<endl;
    endTime = clock();
    cout << "homo matrix calculate time : " <<(double)1000*(endTime - startTime) / CLOCKS_PER_SEC << "ms" << endl;
    
    Mat result;
    rectangle(img1, transPoints[0], transPoints[2], cvScalar(1), 3);
    rectangle(img0, initPoints[0], initPoints[2], cvScalar(1), 3);
    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
    imshow("result", result);
    imwrite("result.jpg",result);
    
    vector<Point2d> rectPts(4),transPts(4);
    /*rectPts[0]=upLeft;
    rectPts[1]=lowLeft;
    rectPts[2]=lowRight;
    rectPts[3]=upRight;*/
    
    for(int j=0; j<4; j++){
        rectPts[j] = initPoints[j];
    }
    //rectPts[0]=Point(108, 185);
    //rectPts[1]=Point(108, 264);
    //rectPts[2]=Point(164, 264);
    //rectPts[3]=Point(164, 185);
    
    
    
    Point points[1][4];
    perspectiveTransform(rectPts, transPts, homo);
    for(int i=0; i<4; i++)
    {
        //cout << transPts[i] << endl;
        points[0][i]=transPts[i];
        transPoints[i]=transPts[i];
    }
    
    const Point* pt[1] = { points[0] };
    int npt[1] = {4};
    //polylines(img2, pt, npt, 1, 1, Scalar(250,0,0)) ;
    polylines(img2, pt, npt, 1, 1, Scalar(1),3);
    //imshow("img2_poly", img2);
    
    Mat transResult(img0.rows,img0.cols + img2.cols+1,img0.type());
    img0.colRange(0,img0.cols).copyTo(transResult.colRange(0,img0.cols));
    img2.colRange(0,img2.cols).copyTo(transResult.colRange(img2.cols+1,transResult.cols));
    imshow("transResult",transResult);
    char num[10];
    sprintf(num, "%d", count);
    string tmp = string(num), name = "transResult" +tmp+".jpg";
    imwrite(name,transResult);
    
    waitKey();
    return homo;
}


int main(int argc, char *argv[])
{
    vector<String> fileName;
    
    /*fileName.push_back("AJ/init.png");
    fileName.push_back("AJ/10.png");
    fileName.push_back("AJ/11.png");
    fileName.push_back("AJ/12.png");
    fileName.push_back("AJ/13.png");
    fileName.push_back("AJ/14.png");
    fileName.push_back("AJ/15.png");
    fileName.push_back("AJ/16.png");*/
    
    fileName.push_back("AJ/init.png");
    fileName.push_back("AJ/8.png");
    fileName.push_back("AJ/7.png");
    fileName.push_back("AJ/6.png");
    fileName.push_back("AJ/5.png");
    fileName.push_back("AJ/4.png");
    fileName.push_back("AJ/3.png");
    fileName.push_back("AJ/2.png");

    
    vector<Point2d> transPoints(4), initPoints(4);
    /*initPoints[0]=Point(100, 180);
    initPoints[1]=Point(100, 270);
    initPoints[2]=Point(170, 270);
    initPoints[3]=Point(170, 180);*/
    
    initPoints[0]=Point(100, 183);
    initPoints[1]=Point(100, 267);
    initPoints[2]=Point(163, 267);
    initPoints[3]=Point(163, 183);
    
    
    
    
    
    for(int i=0; i<4; i++){
        transPoints[i] = initPoints[i];
    }
    clock_t startTime, endTime;
    startTime = clock();
    
    
    Mat homo0 = tranformMatrix(fileName, 0, Mat(), transPoints, initPoints);
    Mat homo1 = tranformMatrix(fileName, 1, homo0, transPoints, initPoints);
    Mat homo2 = tranformMatrix(fileName, 2, homo1, transPoints, initPoints);
    Mat homo3 = tranformMatrix(fileName, 3, homo2, transPoints, initPoints);
    Mat homo4 = tranformMatrix(fileName, 4, homo3, transPoints, initPoints);
    Mat homo5 = tranformMatrix(fileName, 5, homo4, transPoints, initPoints);
    Mat homo6 = tranformMatrix(fileName, 6, homo5, transPoints, initPoints);
    
    endTime = clock();
    cout << "time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    
    //tranformMatrix(fileName, Point(348,335), Point(552,786));
    //jointImage(fileName);
    //selectShoesArea();
    //test();
    return 0;
}
