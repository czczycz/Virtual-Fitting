#include <iostream>
#include <opencv2/opencv.hpp>
#include<string>
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

int main()
{
    
    Mat org = imread("image/90.jpg"),temp1,temp2;
    while (waitKey(30) != 27) {
        org.copyTo(temp1);//用来显示点的坐标以及临时的方框
        namedWindow("img");//定义一个img窗口
        setMouseCallback("img", on_mouse, (void*)&org);//调用回调函数
        if(flag) rectangle(temp1, lclick, mlocation, cv::Scalar((0,0,0), (0,0,0), (0,0,0)));
        putText(temp1,"("+std::to_string(mlocation.x)+","+std::to_string(mlocation.y)+")" , mlocation, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
        imshow("img",temp1);
        
    }  
    imwrite("selected.jpg", temp1);
    for (int i =0; i <4; i++) cout << pts[i] <<endl;
}
