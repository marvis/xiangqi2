#include <highgui.h>  
#include <cv.h>  
#include <math.h>  
#include <string>
using namespace std;

int main(int argc, char** argv)  
{  
	IplImage* src;  
	src = cvLoadImage( argv[1], 0 ); //加载灰度图  

	IplImage* dst = cvCreateImage( cvGetSize( src ), IPL_DEPTH_8U, 1 );  
	IplImage* color_dst = cvCreateImage( cvGetSize( src ), IPL_DEPTH_8U, 3 );  //创建三通道图像  
	CvMemStorage* storage = cvCreateMemStorage(0);  
	CvSeq* lines = 0;  
	cvCanny( src, dst, 50, 100, 3 );  //首先运行边缘检测，结果以灰度图显示（只有边缘）  
	cvCvtColor( dst, color_dst, CV_GRAY2BGR ); //色彩空间转换，将dst转换到另外一个色彩空间即3通道图像  
	lines = cvHoughLines2( dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 80, 30, 10 ); //直接得到直线序列  

	//循环直线序列  
	for( int i = 0; i < lines ->total; i++ )  //lines存储的是直线  
	{  
		CvPoint* line = ( CvPoint* )cvGetSeqElem( lines, i );  //lines序列里面存储的是像素点坐标  
		cvLine( color_dst, line[0], line[1], CV_RGB( 0, 255, 0 ) );  //将找到的直线标记为红色  
		//color_dst是三通道图像用来存直线图像  
	}  
	string filename = argv[1];
	filename = filename + ".lines.png";
	cvSaveImage( filename.c_str(), color_dst );  

	return 0;  
} 
