#include <stdio.h>  
#include "cv.h"  
#include "highgui.h"  
#include <string>
#include <algorithm>
using namespace std;

#define MAX_CORNERS 200  

int main(int argc, char ** argv)  
{  
	int cornersCount=MAX_CORNERS;//得到的角点数目  
	CvPoint2D32f corners[MAX_CORNERS];//输出角点集合  
	vector<int> xcorners, ycorners;
	IplImage *srcImage = 0,*grayImage = 0,*corners1 = 0,*corners2 = 0;  
	int i;  
	CvScalar color = CV_RGB(0,0,255);  
	string filename = argv[1];
	cvNamedWindow("image",1);  

	//Load the image to be processed  
	srcImage = cvLoadImage(filename.c_str(),1);  
	grayImage = cvCreateImage(cvGetSize(srcImage),IPL_DEPTH_8U,1);  

	//copy the source image to copy image after converting the format  
	//复制并转为灰度图像  
	cvCvtColor(srcImage,grayImage,CV_BGR2GRAY);  

	// 找出圆的位置
	IplImage * gaussImage = cvCreateImage(cvGetSize(grayImage), IPL_DEPTH_8U, 1);
	CvMemStorage* storage = cvCreateMemStorage(0);
	cvSmooth(grayImage, gaussImage, CV_GAUSSIAN, 5, 5); // 降噪
	CvSeq* results = cvHoughCircles(  //cvHoughCircles函数需要估计每一个像素梯度的方向，  
			//因此会在内部自动调用cvSobel,而二值边缘图像的处理是比较难的  
			gaussImage,  
			storage,  
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
		xcorners.push_back(cvRound(p[0]));
		ycorners.push_back(cvRound(p[1]));

	}

	//开始画出每个点  
	if (new_cornersCount>0)  
	{  
		for (i=0;i<new_cornersCount;i++)  
		{  
			cvCircle(srcImage,cvPoint((int)(new_corners[i].x),(int)(new_corners[i].y)),  
					2,color,2,CV_AA,0);  
			xcorners.push_back((int)(new_corners[i].x));
			ycorners.push_back((int)(new_corners[i].y));
		}  
	}  

	int nx = 9;
	int min_x = * (min_element(xcorners.begin(), xcorners.end()));
	int max_x = * (max_element(xcorners.begin(), xcorners.end()));
	int max_xstep = (int)((max_x - min_x)/(nx-1.0) + 0.5);
	int min_xstep = 50;
	int bst_x0 = -1, bst_xstep = -1;
	int max_xcount = 0;
	for(int xstep = min_xstep; xstep <= max_xstep; xstep++)
	{
		for(int x0 = min_x; x0 + (nx-1.0)*xstep <= max_x; x0++)
		{
			int xcount = 0;
			for(int i = 0; i < nx; i++)
			{
				int x1 = x0 + i*xstep;
				for(int j = 0; j < xcorners.size(); j++)
				{
					int dst = (xcorners[j] - x1)*(xcorners[j] - x1);
					if(dst <= 8*8) xcount++;
				}
			}
			if(xcount > max_xcount)
			{
				max_xcount = xcount;
				bst_x0 = x0;
				bst_xstep = xstep;
			}
		}
	}	
	
	int ny = 10;
	int min_y = * (min_element(ycorners.begin(), ycorners.end()));
	int max_y = * (max_element(ycorners.begin(), ycorners.end()));
	int max_ystep = (int)((max_y - min_y)/(ny - 1.0) + 0.5);
	int min_ystep = 60;
	int bst_y0 = -1, bst_ystep = -1;
	int max_ycount = 0;
	for(int ystep = min_ystep; ystep <= max_ystep; ystep++)
	{
		int max_ycount2 = 0;
		cout<<ystep<<" : "<<endl;
		for(int y0 = min_y; y0 + (ny-1)*ystep <= max_y; y0++)
		{
			int ycount = 0;
			cout<<"    "<<y0<<" : ";
			for(int i = 0; i < ny; i++)
			{
				int y1 = y0 + i*ystep;
				int ysubcount = 0;
				for(int j = 0; j < ycorners.size(); j++)
				{
					int dst = (ycorners[j] - y1)*(ycorners[j] - y1);
					if(dst <= 10*10)
					{
						ysubcount++;
						ycount++;
					}
				}
				cout<<ysubcount<<" ";
			}
			cout<<"("<<ycount<<")"<<endl;
			if(ycount > max_ycount)
			{
				max_ycount = ycount;
				bst_y0 = y0;
				bst_ystep = ystep;
			}
			if(ycount > max_ycount2) max_ycount2 = ycount;
		}
		cout<<"    max = "<<max_ycount2<<endl;
	}
	cout<<"min_y = "<<min_y<<endl;
	cout<<"max_y = "<<max_y<<endl;
	cout<<"bst_xstep = "<<bst_xstep<<endl;
	cout<<"bst_ystep = "<<bst_ystep<<endl;

	for(int i = 0; i < nx; i++)
	{
		int x = bst_x0 + i*bst_xstep;
		cvLine(srcImage, cvPoint(x, bst_y0), cvPoint(x, bst_y0 + (ny-1)*bst_ystep), cvScalar(255, 0, 0));
	}


	for(int i = 0; i < ny; i++)
	{
		int y = bst_y0 + i*bst_ystep;
		cvLine(srcImage, cvPoint(bst_x0, y), cvPoint(bst_x0 + (nx-1)*bst_xstep, y), cvScalar(0, 255, 0));
	}

	//the font variable    
	CvFont font;    
	double hScale=1;   
	double vScale=1;    
	int lineWidth=2;// 相当于写字的线条    
	// 初始化字体   
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);//初始化字体，准备写到图片上的

	for( int i = 0; i < results->total; i++ )  
	{
		float* p = ( float* )cvGetSeqElem( results, i );  
		//霍夫圆变换  
		CvPoint pt = cvPoint( cvRound( p[0] ), cvRound( p[1] ) );  
		int x1 = cvRound(p[0]);
		int y1 = cvRound(p[1]);
		int xpos = (x1 - bst_x0)/(double)bst_xstep + 0.5 + 1;
		int ypos = (y1 - bst_y0)/(double)bst_ystep + 0.5 + 1;
		ostringstream oss;
		oss<<ypos<<","<<xpos;
		cvPutText(srcImage, oss.str().c_str(), cvPoint(x1-30,y1), &font, CV_RGB(0,255,0));
	}

	cvShowImage("image",srcImage);  
	filename = filename + ".fit.png";
	cvSaveImage(filename.c_str(),srcImage);  

	cvReleaseImage(&srcImage);  
	cvReleaseImage(&grayImage);  
	cvReleaseImage(&gaussImage);  
	cvReleaseImage(&corners1);  
	cvReleaseImage(&corners2);  

	cvWaitKey(0);  
	return 0;  
} 