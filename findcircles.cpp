#include <highgui.h>  
#include <math.h>  
#include <cv.h>  
#include <iostream>
using namespace std;

int main(int argc, char** argv)  
{  
	IplImage* orig = cvLoadImage( argv[1], 1 );  
	IplImage* src = cvLoadImage( argv[1], 0 );  
	IplImage* dst = cvLoadImage( argv[1], 0 );  
	CvMemStorage* storage = cvCreateMemStorage(0);  
	cvSmooth( src, dst, CV_GAUSSIAN, 5, 5 );  //降噪  
	CvSeq* results = cvHoughCircles(  //cvHoughCircles函数需要估计每一个像素梯度的方向，  
			//因此会在内部自动调用cvSobel,而二值边缘图像的处理是比较难的  
			dst,  
			storage,  
			CV_HOUGH_GRADIENT,  
			2,  //累加器图像的分辨率  
			src->width/15,
			100,
			100,
			30, 40
			);  
	printf("total circles = %d\n", results->total);
	for( int i = 0; i < results->total; i++ )  
	{
		float* p = ( float* )cvGetSeqElem( results, i );  
		cout<<p[2]<<" ";
		//霍夫圆变换  
		CvPoint pt = cvPoint( cvRound( p[0] ), cvRound( p[1] ) );  
		cvCircle(  
				orig,  
				pt,  //确定圆心  
				cvRound( p[2] ),  //确定半径  
				CV_RGB( 0x0, 0x0, 0xff )  
				);  //画圆函数  
	}  
	cout<<endl;

	string filename = argv[1];
	filename = filename + ".cirles.png";
	cvSaveImage(filename.c_str(), orig);

	cvClearSeq(results);
	cvReleaseMemStorage(&storage);
	return 0;  
} 
