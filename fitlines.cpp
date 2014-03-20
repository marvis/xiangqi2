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
		//cvCircle(srcImage,  pt, cvRound(2), CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
		xcorners.push_back(cvRound(p[0]));
		ycorners.push_back(cvRound(p[1]));

	}

	//开始画出每个点  
	if (new_cornersCount>0)  
	{  
		for (i=0;i<new_cornersCount;i++)  
		{  
			//cvCircle(srcImage,cvPoint((int)(new_corners[i].x),(int)(new_corners[i].y)), 2,color,2,CV_AA,0);  
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
					if(dst <= 10*10) xcount++;
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
		for(int y0 = min_y; y0 + (ny-1)*ystep <= max_y; y0++)
		{
			int ycount = 0;
			for(int i = 0; i < ny; i++)
			{
				int y1 = y0 + i*ystep;
				for(int j = 0; j < ycorners.size(); j++)
				{
					int dst = (ycorners[j] - y1)*(ycorners[j] - y1);
					if(dst <= 10*10)
						ycount++;
				}
			}
			if(ycount > max_ycount)
			{
				max_ycount = ycount;
				bst_y0 = y0;
				bst_ystep = ystep;
			}
		}
	}

	CvScalar color_tab[12];
	color_tab[0] = CV_RGB(255,0,0);
	color_tab[1] = CV_RGB(0,255,0);
	color_tab[2] = CV_RGB(0,0,255);
	color_tab[3] = CV_RGB(0,255,255);
	color_tab[4] = CV_RGB(255,0,255);
	color_tab[5] = CV_RGB(255,255,0);
	color_tab[6] = CV_RGB(0,100,255);
	color_tab[7] = CV_RGB(255,155, 0);
	color_tab[8] = CV_RGB(100,0,255);
	color_tab[9] = CV_RGB(100,255,0);
	color_tab[10] = CV_RGB(255,0,100);
	color_tab[11] = CV_RGB(255,100,0);

	//the font variable    
	CvFont font;    
	double hScale=1;   
	double vScale=1;    
	int lineWidth=2;// 相当于写字的线条    
	// 初始化字体   
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);//初始化字体，准备写到图片上的

	// opencv line fit
	vector<int> xclusters(xcorners.size(), 0);
	int count = 0;
	for(int i = 0; i < nx; i++)
	{
		int x1 = bst_x0 + i * bst_xstep;
		for(int j = 0; j < xcorners.size(); j++)
		{
			int dst = (xcorners[j]-x1)*(xcorners[j]-x1);
			if(dst <= 10*10)
			{
				if(xclusters[j] == 0)
				{
					xclusters[j] = i+1;
					count++;
				}
				else
				{
					cout<<"impossible"<<endl;
					return -1;
				}
			}
		}
	}

	CvPoint* x_points = (CvPoint*)malloc( count* sizeof(x_points[0]));
	CvMat x_pointMat = cvMat( 1, count, CV_32SC2, x_points ); //点集, 存储count个随机点points
	float x_params[4]; //输出的直线参数。2D 拟合情况下，它是包含 4 个浮点数的数组 (vx, vy, x0, y0)  
	int j = 0;
	vector<int> sumY(nx, 0);
	vector<int> countY(nx, 0);
	for(int i = 0; i < xclusters.size(); i++)
	{
		int clusterId = xclusters[i];
		if(clusterId > 0)
		{
			x_points[j].x = xcorners[i] - (clusterId-1) * bst_xstep;
			x_points[j].y = ycorners[i];
			sumY[clusterId-1] += ycorners[i];
			countY[clusterId-1]++;
			j++;
		}
	}

	cvFitLine( &x_pointMat, CV_DIST_L1, 1, 0.001, 0.001, x_params ); // find the optimal line 曲线拟合
	double vx = x_params[0];
	double vy = x_params[1];
	double x0 = x_params[2];
	double y0 = x_params[3];
	for(int i = 0; i < nx; i++)
	{
		CvScalar color = color_tab[i+1];
		double mx = x0 + i * bst_xstep;
		double my = sumY[i]/countY[i];
		//x = mx + vx*(y-my)/vy;
		int leftx = fabs(vy) > 0.00000001 ? mx + vx * (0 - my)/vy : mx;
		int rightx = fabs(vy) > 0.00000001 ? mx + vx * (srcImage->height -1 - my)/vy : mx;
		cvLine(srcImage, cvPoint(leftx, 0), cvPoint(mx, my), color);
		color.val[0] = 255 - color.val[0];
		color.val[1] = 255 - color.val[1];
		color.val[2] = 255 - color.val[2];
		cvLine(srcImage, cvPoint(mx,my), cvPoint(rightx, srcImage->height-1), color);
	}

	vector<int> yclusters(ycorners.size(), 0);
	count = 0;
	for(int i = 0; i < ny; i++)
	{
		int y1 = bst_y0 + i * bst_ystep;
		for(int j = 0; j < ycorners.size(); j++)
		{
			int dst = (ycorners[j]-y1)*(ycorners[j]-y1);
			if(dst <= 10*10)
			{
				if(yclusters[j] == 0)
				{
					yclusters[j] = i+1;
					count++;
				}
				else
				{
					cout<<"impossible"<<endl;
					return -1;
				}
			}
		}
	}

	CvPoint* y_points = (CvPoint*)malloc( count* sizeof(y_points[0]));
	CvMat y_pointMat = cvMat( 1, count, CV_32SC2, y_points ); //点集, 存储count个随机点points
	float y_params[4]; //输出的直线参数。2D 拟合情况下，它是包含 4 个浮点数的数组 (vx, vy, x0, y0)  
	j = 0;
	vector<int> sumX(ny, 0);
	vector<int> countX(ny, 0);
	for(int i = 0; i < yclusters.size(); i++)
	{
		int clusterId = yclusters[i];
		if(clusterId > 0)
		{
			y_points[j].x = xcorners[i];
			y_points[j].y = ycorners[i] - (clusterId-1) * bst_ystep;
			sumX[clusterId-1] += xcorners[i];
			countX[clusterId-1]++;
			j++;
		}
	}

	cvFitLine( &y_pointMat, CV_DIST_L1, 1, 0.001, 0.001, y_params ); // find the optimal line 曲线拟合
	vx = y_params[0];
	vy = y_params[1];
	x0 = y_params[2];
	y0 = y_params[3];
	for(int i = 0; i < ny; i++)
	{
		CvScalar color = color_tab[i+1];
		double mx = sumX[i]/countX[i];
		double my = y0 + i * bst_ystep;
		//y = my + vy*(x-mx)/vx;
		int topy = fabs(vx) > 0.00000001 ? my + vy * (0 - mx)/vx : my;
		int boty = fabs(vx) > 0.00000001 ? my + vy * (srcImage->width -1 - mx)/vx : my;
		cvLine(srcImage, cvPoint(0, topy), cvPoint(mx, my), color);
		color.val[0] = 255 - color.val[0];
		color.val[1] = 255 - color.val[1];
		color.val[2] = 255 - color.val[2];
		cvLine(srcImage, cvPoint(mx,my), cvPoint(srcImage->width-1, boty), color);
	}

	/*
	for(int i = 0; i < xcorners.size(); i++)
	{
		int x = xcorners[i];
		int y = ycorners[i];
		int clusterId = xclusters[i];
		CvScalar color = color_tab[clusterId];
		cvCircle(srcImage,cvPoint(x,y), 2,color,2,CV_AA,0);  
		if(clusterId > 0)
		{
			ostringstream oss;
			oss<<clusterId;
			cvPutText(srcImage, oss.str().c_str(), cvPoint(x-20,y+15), &font, CV_RGB(0,255,255));
		}
	}*/	

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
		cvPutText(srcImage, oss.str().c_str(), cvPoint(x1-30,y1-15), &font, CV_RGB(0,255,0));
	}

	filename = filename + ".fit.png";
	cvSaveImage(filename.c_str(),srcImage);  

	cvReleaseImage(&srcImage);  
	cvReleaseImage(&grayImage);  
	cvReleaseImage(&gaussImage);  
	cvReleaseImage(&corners1);  
	cvReleaseImage(&corners2);  

	cvClearSeq(results);
	cvReleaseMemStorage(&storage);
	return 0;  
} 
