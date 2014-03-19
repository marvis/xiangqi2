#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；
#include <cmath>
#include <iostream>
#include <stack>

using namespace cv;
using namespace std;

IplImage * cropImage(IplImage * src, int x, int y, int width, int height)
{
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}

int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("No input image\n");
		return -1;
	}
	IplImage * image0 = cvLoadImage(argv[1], 0); // load as gray image
	IplImage * image1 = cvLoadImage(argv[1], 1); // load as gray image
	if(!image0 || !image1)
	{
		printf( "No image data \n" );
		return -1;
	}
	string filename0 = argv[1];
	string filename1;
	int width = image0->width;
	int height = image0->height;
	int nchannels = image0->nChannels;


	// find maximum component first time
	IplImage * binImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
			if(val > 30) 
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
			}
			else 
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
			}
		}
	}

	filename1 = filename0 + ".bin1.png";
	cvSaveImage(filename1.c_str(), binImg);

	int max_color = 256*256*256 - 1;
	int color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(binImg, cvPoint(w,h), CV_RGB(hig, mid, low));
					color++;
				}
			}
			else
			{
				cout<<"Too many connected areas. "<<endl;
				return -1;
			}
		}
	}
	//cout<<"color = "<<color<<endl;
	int colorNum = color;

	vector<int>colorsum(colorNum, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= colorNum)
			{
				cout<<"invalid rgb"<<endl;
				return -1;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}

	int maxcolorLabel = max_element(colorsum.begin(), colorsum.end()) - colorsum.begin();
	//printf("maxcolorLabel = %d\n", maxcolorLabel);

	int min_w = width-1, max_w = 0;
	int min_h = height-1, max_h = 0;
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val == maxcolorLabel)
			{
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 255;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 255;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 255;  
				min_w = (w < min_w) ? w : min_w;
				max_w = (w > max_w) ? w : max_w;
				min_h = (h < min_h) ? h : min_h;
				max_h = (h > max_h) ? h : max_h;
			}
			else
			{  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 0;  
			}
		}
	}

	filename1 = filename0 + ".bin2.png";
	cvSaveImage(filename1.c_str(), binImg);

	IplConvKernel *element = cvCreateStructuringElementEx(10, 10, 0, 0, CV_SHAPE_ELLIPSE);
	cvMorphologyEx(binImg, binImg, NULL, element, CV_MOP_CLOSE);//关运算，填充内部的细线

	filename1 = filename0 + ".bin3.png";
	cvSaveImage(filename1.c_str(), binImg);

	IplImage * binImg2 = cropImage(binImg, min_w, min_h, (max_w-min_w+1), (max_h-min_h+1));
	cvReleaseImage(&binImg);
	
	binImg = binImg2;
	width = binImg->width;
	height = binImg->height;

	// find maximum component second time
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			unsigned char b = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i);
			unsigned char g = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1);
			unsigned char r = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2);

			CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255 - b;
			CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255 - g;
			CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255 - r;
		}
	}

	filename1 = filename0 + ".bin4.png";
	cvSaveImage(filename1.c_str(), binImg);

	max_color = 256*256*256 - 1;
	color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(binImg, cvPoint(w,h), CV_RGB(hig, mid, low));
					color++;
				}
			}
			else
			{
				cout<<"Too many connected areas. "<<endl;
				return -1;
			}
		}
	}
	//cout<<"color = "<<color<<endl;
	colorNum = color;

	colorsum.resize(colorNum, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= colorNum)
			{
				cout<<"invalid rgb"<<endl;
				return -1;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}

	maxcolorLabel = max_element(colorsum.begin(), colorsum.end()) - colorsum.begin();
	//printf("maxcolorLabel = %d\n", maxcolorLabel);

	double min_tl_dist = width*width+height*height;
	double min_tr_dist = min_tl_dist;
	double min_bl_dist = min_tl_dist;
	double min_br_dist = min_tl_dist;
	int mtlx, mtly;
	int mtrx, mtry;
	int mblx, mbly;
	int mbrx, mbry;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val == maxcolorLabel)
			{
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 255;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 255;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 255;  
				double tl_dist = w*w + h*h;
				double tr_dist = (width-1-w)*(width-1-w) + h*h;
				double bl_dist = w*w + (height-1-h)*(height-1-h);
				double br_dist = (width-1-w)*(width-1-w) + (height-1-h)*(height-1-h);
				if(tl_dist < min_tl_dist) {mtlx = w; mtly = h; min_tl_dist = tl_dist;}
				if(tr_dist < min_tr_dist) {mtrx = w; mtry = h; min_tr_dist = tr_dist;}
				if(bl_dist < min_bl_dist) {mblx = w; mbly = h; min_bl_dist = bl_dist;}
				if(br_dist < min_br_dist) {mbrx = w; mbry = h; min_br_dist = br_dist;}
			}
			else
			{  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 0;  
			}
		}
	}

	double mid_w = width/2.0;//(mtlx+mtrx+mblx+mbrx)/4.0;
	double mid_h = height/2.0;//(mtly+mtry+mbly+mbry)/4.0;
	double max_tl_dist = 0;
	double max_tr_dist = 0;
	double max_bl_dist = 0;
	double max_br_dist = 0;
	int Mtlx, Mtly;
	int Mtrx, Mtry;
	int Mblx, Mbly;
	int Mbrx, Mbry;

	IplImage * tmpImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	cvCvtColor(binImg, tmpImg, CV_BGR2GRAY);

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val >0)
			{
				double dist = (w-mid_w)*(w-mid_w) + (h-mid_h)*(h-mid_h);
				if(w < mid_w && h < mid_h && dist > max_tl_dist) {Mtlx = w; Mtly = h; max_tl_dist = dist;}
				if(w >= mid_w && h < mid_h && dist > max_tr_dist) {Mtrx = w; Mtry = h; max_tr_dist = dist;}
				if(w < mid_w && h >= mid_h && dist > max_bl_dist) {Mblx = w; Mbly = h; max_bl_dist = dist;}
				if(w >= mid_w && h >= mid_h && dist > max_br_dist) {Mbrx = w; Mbry = h; max_br_dist = dist;}
			}
		}
	}
	
	//两条对角线的方程
	//(y-y1)/(x-x1) = (y2-y1)/(x2-x1) = k
	double k1 = (Mtry - Mbly)/(double)(Mtrx - Mblx); // k1 < 0
	double b1 = Mbly - k1 * Mblx;
	double k2 = (Mbry - Mtly)/(double)(Mbrx - Mtlx); // k2 > 0
	double b2 = Mbry - k2 * Mbrx;
	double cross_x = -(b2 - b1)/(k2-k1);
	double cross_y = k1*cross_x + b1;

	//找出图像外轮廓
	CvMemStorage* storage= cvCreateMemStorage(0);
	CvSeq* contour= 0;//可动态增长元素序列
	cvFindContours(tmpImg,//二值化图像
			storage,//轮廓存储容器
			&contour,//指向第一个轮廓输出的指针
			sizeof(CvContour),//序列头大小
			CV_RETR_EXTERNAL,//CV_RETR_CCOMP 内外轮廓都检测;CV_RETR_EXTERNAL：只检索最外面的轮廓
			CV_CHAIN_APPROX_SIMPLE//压缩水平垂直和对角分割，即函数只保留末端的象素点
			//CV_CHAIN_CODE//CV_CHAIN_APPROX_NONE
			);//函数cvFindContours从二值图像中检索轮廓，并返回检测到的轮廓的个数

	cvSetZero(binImg);
	//cvReleaseImage(&binImg);
	//binImg = cropImage(image1, min_w, min_h, max_w-min_w+1, max_h-min_h+1);

	// there should be only one contour returned
	/*if(0 && contour) // 将边界点进行多边形拟合, 不过效果不怎么好
	{
		CvSeq* cont, *mcont;
		CvTreeNodeIterator iterator;
		cvInitTreeNodeIterator (&iterator, contour,  1); 
		CvMemStorage* storage1 = cvCreateMemStorage(0);
		while (0 != (cont = (CvSeq*)cvNextTreeNode (&iterator)))  
		{  
			mcont = cvApproxPoly (cont, sizeof(CvContour), storage1, CV_POLY_APPROX_DP, cvContourPerimeter(cont)*0.02,0);  
			cvDrawContours (binImg, mcont, CV_RGB(255,0,0),CV_RGB(0,0,100),1,2,8,cvPoint(0,0));  
		}
	}*/

	vector<CvPoint> leftPoints, rightPoints, topPoints, bottomPoints;
	for(int i = 0; i < contour-> total; i++)    // 提取一个轮廓的所有坐标点  
	{  
		CvPoint *pt = (CvPoint*) cvGetSeqElem(contour, i);   // 得到一个轮廓中一个点的函数cvGetSeqElem  
		double flag1 = pt->y - k1*pt->x - b1;
		double flag2 = pt->y - k2*pt->x - b2;
		if(flag1 < 0 && flag2 > 0) 
		{
			leftPoints.push_back(*pt);
			cvCircle(binImg, *pt , 2, CV_RGB( 0x0, 0xff, 0xff ), 2, CV_AA, 0);  //画圆函数  
		}
		if(flag1 > 0 && flag2 < 0)
		{
			rightPoints.push_back(*pt);
			cvCircle(binImg, *pt , 2, CV_RGB( 0xff, 0x0, 0xff ), 2, CV_AA, 0);  //画圆函数  
		}
		if(flag1 < 0 && flag2 < 0) 
		{
			topPoints.push_back(*pt);
			cvCircle(binImg, *pt , 2, CV_RGB( 0xff, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
		}
		if(flag1> 0 && flag2 > 0)
		{
			if(pt->x > 0.75*Mblx + 0.25*Mbrx) // 由于有一个菜单在左下角
			{
				bottomPoints.push_back(*pt);
				cvCircle(binImg, *pt , 2, CV_RGB( 128, 128, 0 ), 2, CV_AA, 0);  //画圆函数  
			}
		}
		//cvCircle(binImg, *pt , 2, CV_RGB( 0xff, 0xff, 0xff ), 2, CV_AA, 0);  //画圆函数  
	}
	cout<<"bottomPoints.size() = "<<bottomPoints.size()<<endl;

	CvPoint * leftpoints = (CvPoint*)malloc(leftPoints.size() * sizeof(CvPoint));
	CvMat leftPointMat = cvMat(1, leftPoints.size(), CV_32SC2, leftpoints);
	CvPoint * rightpoints = (CvPoint*)malloc(rightPoints.size() * sizeof(CvPoint));
	CvMat rightPointMat = cvMat(1, rightPoints.size(), CV_32SC2, rightpoints);
	CvPoint * toppoints = (CvPoint*)malloc(topPoints.size() * sizeof(CvPoint));
	CvMat topPointMat = cvMat(1, topPoints.size(), CV_32SC2, toppoints);
	CvPoint * bottompoints = (CvPoint*)malloc(bottomPoints.size() * sizeof(CvPoint));
	CvMat bottomPointMat = cvMat(1, bottomPoints.size(), CV_32SC2, bottompoints);
	for(int i = 0; i < leftPoints.size(); i++)
	{
		leftpoints[i].x = leftPoints[i].x;
		leftpoints[i].y = leftPoints[i].y;
	}
	for(int i = 0; i < rightPoints.size(); i++)
	{
		rightpoints[i].x = rightPoints[i].x;
		rightpoints[i].y = rightPoints[i].y;
	}
	for(int i = 0; i < topPoints.size(); i++)
	{
		toppoints[i].x = topPoints[i].x;
		toppoints[i].y = topPoints[i].y;
	}
	for(int i = 0; i < bottomPoints.size(); i++)
	{
		bottompoints[i].x = bottomPoints[i].x;
		bottompoints[i].y = bottomPoints[i].y;
	}

	float leftParams[4], rightParams[4], topParams[4], bottomParams[4];
	cvFitLine(&leftPointMat, CV_DIST_L1, 1, 0.001, 0.001, leftParams);
	double left_vx = leftParams[0];
	double left_vy = leftParams[1];
	double left_x0 = leftParams[2];
	double left_y0 = leftParams[3];
	double left_y1 = Mtly;
	double left_y2 = Mbly;
	double left_x1 = left_x0;
	double left_x2 = left_x0;
	if(fabs(left_vy) > 0.000001)
	{
		//x = x0 + vx/vy * (y - y0)
		left_x1 = left_x0 + left_vx/left_vy * (left_y1 - left_y0);
		left_x2 = left_x0 + left_vx/left_vy * (left_y2 - left_y0);
	}
	cvLine(binImg, cvPoint(left_x1, left_y1), cvPoint(left_x2, left_y2), CV_RGB(0xff, 0x0, 0x0));
	
	cvFitLine(&rightPointMat, CV_DIST_L1, 1, 0.001, 0.001, rightParams);
	double right_vx = rightParams[0];
	double right_vy = rightParams[1];
	double right_x0 = rightParams[2];
	double right_y0 = rightParams[3];
	double right_y1 = Mtry;
	double right_y2 = Mbry;
	double right_x1 = right_x0;
	double right_x2 = right_x0;
	if(fabs(right_vy) > 0.000001)
	{
		//x = x0 + vx/vy * (y - y0)
		right_x1 = right_x0 + right_vx/right_vy * (right_y1 - right_y0);
		right_x2 = right_x0 + right_vx/right_vy * (right_y2 - right_y0);
	}
	cvLine(binImg, cvPoint(right_x1, right_y1), cvPoint(right_x2, right_y2), CV_RGB(0x0, 0xff, 0x0));

	cvFitLine(&topPointMat, CV_DIST_L1, 1, 0.001, 0.001, topParams);
	double top_vx = topParams[0];
	double top_vy = topParams[1];
	double top_x0 = topParams[2];
	double top_y0 = topParams[3];
	double top_x1 = Mtlx;
	double top_x2 = Mtrx;
	double top_y1 = top_y0;
	double top_y2 = top_y0;
	if(fabs(top_vx) > 0.000001)
	{
		//y = y0 + vy/vx * (x - x0)
		top_y1 = top_y0 + top_vy/top_vx * (top_x1 - top_x0);
		top_y2 = top_y0 + top_vy/top_vx * (top_x2 - top_x0);
	}
	cvLine(binImg, cvPoint(top_x1, top_y1), cvPoint(top_x2, top_y2), CV_RGB(0x0, 0x0, 0xff));

	cvFitLine(&bottomPointMat, CV_DIST_L1, 1, 0.001, 0.001, bottomParams);
	double bottom_vx = bottomParams[0];
	double bottom_vy = bottomParams[1];
	double bottom_x0 = bottomParams[2];
	double bottom_y0 = bottomParams[3];
	double bottom_x1 = Mblx;
	double bottom_x2 = Mbrx;
	double bottom_y1 = bottom_y0;
	double bottom_y2 = bottom_y0;
	if(fabs(bottom_vx) > 0.000001)
	{
		//y = y0 + vy/vx * (x - x0)
		bottom_y1 = bottom_y0 + bottom_vy/bottom_vx * (bottom_x1 - bottom_x0);
		bottom_y2 = bottom_y0 + bottom_vy/bottom_vx * (bottom_x2 - bottom_x0);
	}
	cvLine(binImg, cvPoint(bottom_x1, bottom_y1), cvPoint(bottom_x2, bottom_y2), CV_RGB(127, 127, 255));

	cvCircle(binImg, cvPoint(Mtlx, Mtly) , cvRound(15), CV_RGB( 0xff, 0x0, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint(Mtrx, Mtry) , cvRound(15), CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint(Mblx, Mbly) , cvRound(15), CV_RGB( 0xff, 0x0, 0xff ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint(Mbrx, Mbry) , cvRound(15), CV_RGB( 0xff, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint((int)(cross_x+0.5), (int)(cross_y+0.5)) , cvRound(15), CV_RGB( 0x0, 0xff, 0xff), 2, CV_AA, 0);  //画圆函数  
	cout<<"cross_x = "<<cross_x<<", cross_y = "<<cross_y<<endl;

	cvLine(binImg, cvPoint(Mtlx, Mtly), cvPoint(Mbrx, Mbry), CV_RGB(0xff, 0x0, 0x0));
	cvLine(binImg, cvPoint(Mtrx, Mtry), cvPoint(Mblx, Mbly), CV_RGB(0xff, 0x0, 0x0));

	filename1 = filename0 + ".bin6.png";
	cvSaveImage(filename1.c_str(), binImg);

	cvReleaseImage(&image0);
	cvReleaseImage(&image1);
	cvReleaseImage(&binImg);
	cvReleaseImage(&tmpImg);

	cvClearSeq(contour);
	cvReleaseMemStorage(&storage);
	return 0;

}
