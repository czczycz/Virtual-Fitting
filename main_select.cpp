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

bool flag = false;
Point lclick = Point(-1,-1);
Point mlocation = Point(-1, -1);
int pts[4] = {0,0,0,0};
void on_mouse(int event, int x, int y, int flags, void *ustc)
//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    Mat& image = *(cv::Mat*) ustc;//这样就可以传递Mat信息了
    char temp[16];
    
    if(event == CV_EVENT_LBUTTONDOWN)//按下左键
    {
        sprintf(temp, "(%d,%d)", x, y);
        putText(image, temp, Point(x, y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0,0,0,255));
        flag = true;
        lclick = Point(x, y);
        pts[0] = x;
        pts[1] = y;
    }
    else if (event == CV_EVENT_MOUSEMOVE)//移动鼠标
    {
        mlocation = Point(x, y);
        if (flag)
        { }
    }
    else if (event == EVENT_LBUTTONUP)
    {
        flag = false;
        sprintf(temp, "(%d,%d)", x, y);
        putText(image, temp, Point(x, y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 0, 255));
        //调用函数进行绘制
        cv::rectangle(image,lclick, mlocation, cv::Scalar((0,0,0), (0,0,0), (0,0,0)));//随机颜色
        pts[2] = x;
        pts[3] = y;
        
    }
}

void tranformMatrix(vector<String>& fileName, Point upLeft, Point lowRight)
{
    clock_t startTime, endTime;
    
    Mat img1=imread(fileName[0], IMREAD_GRAYSCALE);
    Mat img2=imread(fileName[1], IMREAD_GRAYSCALE);
    
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
    rectangle(img1, upLeft, lowRight, cvScalar(1), 3);
    drawMatches(img1, keyImg1, img2, keyImg2, bestMatches, result);
    imshow("result", result);
    imwrite("result.jpg",result);
    
    vector<Point2d> rectPts(4), transPts(4);
    rectPts[0]=upLeft;
    rectPts[1]=Point(upLeft.x, lowRight.y);
    rectPts[2]=lowRight;
    rectPts[3]=Point(lowRight.x, upLeft.y);
    
    Point points[1][4];
    perspectiveTransform(rectPts, transPts, homo);
    for(int i=0; i<4; i++)
    {
        //cout << transPts[i] << endl;
        points[0][i]=transPts[i];
    }
    
    const Point* pt[1] = { points[0] };
    int npt[1] = {4};
    //polylines(img2, pt, npt, 1, 1, Scalar(250,0,0)) ;
    polylines(img2, pt, npt, 1, 1, Scalar(1),3);
    //imshow("img2_poly", img2);
    
    Mat transResult(img1.rows,img1.cols + img2.cols+1,img1.type());
    img1.colRange(0,img1.cols).copyTo(transResult.colRange(0,img1.cols));
    img2.colRange(0,img2.cols).copyTo(transResult.colRange(img2.cols+1,transResult.cols));
    imshow("tansResult",transResult);
    imwrite("tansResult.jpg",transResult);
    
    waitKey();
    return;
}

int main(int argc, char *argv[])
{
    vector<String> fileName;
    fileName.push_back("image/90.jpg");
    fileName.push_back("image/105.jpg");
    
    Mat org = imread("image/90.jpg"),temp1,temp2;
    while (waitKey(30) != 27) {
        org.copyTo(temp1);//用来显示点的坐标以及临时的方框
        namedWindow("img");//定义一个img窗口
        setMouseCallback("img", on_mouse, (void*)&org);//调用回调函数
        if(flag) rectangle(temp1, lclick, mlocation, cv::Scalar(1));
        putText(temp1,"("+std::to_string(mlocation.x)+","+std::to_string(mlocation.y)+")" , mlocation, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
        imshow("img",temp1);
        
    }
    imwrite("selected.jpg", temp1);
    for (int i =0; i <4; i++) cout << pts[i] <<endl;
    
    //Point(pts[0], pts[1])必须在Point(pts[2], pts[3])左上方
    tranformMatrix(fileName, Point(pts[0], pts[1]), Point(pts[2], pts[3]));
    
    return 0;
}














