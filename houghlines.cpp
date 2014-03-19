#include <stdio.h>  
#include "cv.h"  
#include "highgui.h"  
#include <string>
using namespace std;

#define MAX_CORNERS 200  

int main(int argc, char ** argv)  
{  
	int cornersCount=MAX_CORNERS;//得到的角点数目  
	CvPoint2D32f corners[MAX_CORNERS];//输出角点集合  
	IplImage *srcImage = 0,*grayImage = 0,*corners1 = 0,*corners2 = 0;  
	int i;  
	CvScalar color = CV_RGB(0,0,255);  
	string filename = argv[1];

	//Load the image to be processed  
	srcImage = cvLoadImage(filename.c_str(),1);  
	grayImage = cvCreateImage(cvGetSize(srcImage),IPL_DEPTH_8U,1);  

	//copy the source image to copy image after converting the format  
	//复制并转为灰度图像  
	cvCvtColor(srcImage,grayImage,CV_BGR2GRAY);  

	// 找出圆的位置
	IplImage * gaussImage = cvCreateImage(cvGetSize(grayImage), IPL_DEPTH_8U, 1);
	CvMemStorage* storage1 = cvCreateMemStorage(0);
	cvSmooth(grayImage, gaussImage, CV_GAUSSIAN, 5, 5); // 降噪
	CvSeq* results = cvHoughCircles(  //cvHoughCircles函数需要估计每一个像素梯度的方向，  
			//因此会在内部自动调用cvSobel,而二值边缘图像的处理是比较难的  
			gaussImage,  
			storage1,  
			CV_HOUGH_GRADIENT,  
			2,  //累加器图像的分辨率  
			srcImage->width/15,
			100,
			100,
			30, 40
			);  
	//printf("total circles = %d\n", results->total);	

	//create empty images os same size as the copied images  
	//两幅临时32位浮点图像，cvGoodFeaturesToTrack会用到  
	corners1 = cvCreateImage(cvGetSize(srcImage),IPL_DEPTH_32F,1);  
	corners2 = cvCreateImage(cvGetSize(srcImage),IPL_DEPTH_32F,1);  

	int min_dist = 50;
	cvGoodFeaturesToTrack(grayImage,corners1,  
			corners2,corners,  
			&cornersCount,0.05,  
			min_dist,//角点的最小距离是30  
			0,//整个图像  
			3,0,0.4);  
	//默认值  
	//printf("num corners found: %d/n",cornersCount);  
	
	// 去除与圆心距离近的点
	CvPoint2D32f new_corners[MAX_CORNERS];//输出角点集合  
	int new_cornersCount = 0;
	for(int i = 0; i < cornersCount; i++)
	{
		float x1 = (float)(corners[i].x);
		float y1 = (float)(corners[i].y);
		bool isok = true;
		for(int j = 0; j < results->total; j++)
		{
			float* p = ( float* )cvGetSeqElem( results, j );  
			float x2 = p[0];
			float y2 = p[1];
			if((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) < min_dist*min_dist)
			{
				isok = false;
				break;
			}
		}
		if(isok) 
		{
			new_corners[new_cornersCount].x = x1;
			new_corners[new_cornersCount].y = y1;
			new_cornersCount++;
		}
	}

	IplImage * dst = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
	cvZero(dst);

	// 画圆圈
	for( int i = 0; i < results->total; i++ )  
	{
		float* p = ( float* )cvGetSeqElem( results, i );  
		//霍夫圆变换  
		CvPoint pt = cvPoint( cvRound( p[0] ), cvRound( p[1] ) );  
		/*cvCircle(  
				srcImage,  
				pt,  //确定圆心  
				cvRound( p[2] ),  //确定半径  
				CV_RGB( 0x0, 0x0, 0xff )  
				);  //画圆函数  
		*/
		cvCircle(  
				srcImage,  
				pt,  //确定圆心  
				cvRound(2),  //确定半径  
				CV_RGB( 0x0, 0xff, 0x0 ),
				2, CV_AA, 0
				);  //画圆函数  
		
		CV_IMAGE_ELEM(dst, unsigned char, cvRound(p[1]), cvRound(p[0])) = 255;
	}

	//开始画出每个点  
	if (new_cornersCount>0)  
	{  
		for (i=0;i<new_cornersCount;i++)  
		{  
			cvCircle(srcImage,cvPoint((int)(new_corners[i].x),(int)(new_corners[i].y)),  
					2,color,2,CV_AA,0);  
			CV_IMAGE_ELEM(dst, unsigned char, (int)(new_corners[i].y), (int)(new_corners[i].x)) = 255;
		}  
	}  
	
	IplImage* color_dst = cvCreateImage( cvGetSize( dst ), IPL_DEPTH_8U, 3 );  //创建三通道图像  
	CvMemStorage* storage2 = cvCreateMemStorage(0);  
	CvSeq* lines = 0;  
	cvCvtColor( dst, color_dst, CV_GRAY2BGR ); //色彩空间转换，将dst转换到另外一个色彩空间即3通道图像  
	lines = cvHoughLines2( dst, storage2, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 5, 3, 10 ); //直接得到直线序列  

	//循环直线序列  
	for( int i = 0; i < lines ->total; i++ )  //lines存储的是直线  
	{  
		CvPoint* line = ( CvPoint* )cvGetSeqElem( lines, i );  //lines序列里面存储的是像素点坐标  
		cvLine( color_dst, line[0], line[1], CV_RGB( 0, 255, 0 ) );  //将找到的直线标记为红色  
		//color_dst是三通道图像用来存直线图像  
	}

	cvNamedWindow("hough",1);  
	cvShowImage("hough",color_dst);  

	cvNamedWindow("image",1);  
	cvShowImage("image",srcImage);  

	filename = filename + ".hough.png";
	cvSaveImage(filename.c_str(),color_dst);  

	cvReleaseImage(&srcImage);  
	cvReleaseImage(&grayImage);  
	cvReleaseImage(&gaussImage);  
	cvReleaseImage(&corners1);  
	cvReleaseImage(&corners2); 
	cvReleaseImage(&dst);
	cvReleaseImage(&color_dst);

	cvClearSeq(results);
	cvClearSeq(lines);
	cvReleaseMemStorage(&storage1);
	cvReleaseMemStorage(&storage2);
	cvWaitKey(0);  
	return 0;  
} 
