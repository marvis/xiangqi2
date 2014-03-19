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
	if(!image0)
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
	cout<<"color = "<<color<<endl;
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
	printf("maxcolorLabel = %d\n", maxcolorLabel);

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
	cout<<"color = "<<color<<endl;
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
	printf("maxcolorLabel = %d\n", maxcolorLabel);

	double min_tl_dist = width*width+height*height;
	double min_tr_dist = min_tl_dist;
	double min_bl_dist = min_tl_dist;
	double min_br_dist = min_tl_dist;
	int tlx, tly;
	int trx, _try;
	int blx, bly;
	int brx, bry;

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
				if(tl_dist < min_tl_dist) {tlx = w; tly = h; min_tl_dist = tl_dist;}
				if(tr_dist < min_tr_dist) {trx = w; _try = h; min_tr_dist = tr_dist;}
				if(bl_dist < min_bl_dist) {blx = w; bly = h; min_bl_dist = bl_dist;}
				if(br_dist < min_br_dist) {brx = w; bry = h; min_br_dist = br_dist;}
			}
			else
			{  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 0;  
			}
		}
	}
	
	cvCircle(binImg, cvPoint(tlx, tly) , cvRound(15), CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint(trx, _try) , cvRound(15), CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint(blx, bly) , cvRound(15), CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	cvCircle(binImg, cvPoint(brx, bry) , cvRound(15), CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  

	filename1 = filename0 + ".bin5.png";
	cvSaveImage(filename1.c_str(), binImg);
	
    cvReleaseImage(&image0);
	cvReleaseImage(&binImg);
	return 0;

}
