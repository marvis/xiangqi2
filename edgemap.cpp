#include <highgui.h>  
#include <cv.h>  
#include <math.h>  
#include <string>
using namespace std;

int main(int argc, char** argv)  
{  
	IplImage* src;  
	string filename = argv[1];
	src = cvLoadImage( argv[1], 0 ); //加载灰度图  

	IplImage* edge = cvCreateImage( cvGetSize( src ), IPL_DEPTH_8U, 1 );  
	cvCanny( src, edge, 50, 100, 3 );  //首先运行边缘检测，结果以灰度图显示（只有边缘）  

	cvNamedWindow( "Hough", 0 );  
	cvShowImage( "Hough", edge );  
	cvWaitKey(0);
	filename = filename + ".edge.png";
	cvSaveImage(filename.c_str(), edge);
	return 0;  
} 
