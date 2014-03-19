#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；
#include <cmath>
#include <iostream>
#include <stack>

using namespace cv;
using namespace std;
#define DEBUG 0

void rotateImage(IplImage* img, IplImage *img_rotate,int degree)  
{  
    //旋转中心为图像中心  
    CvPoint2D32f center;    
    center.x=float (img->width/2.0+0.5);  
    center.y=float (img->height/2.0+0.5);  
    //计算二维旋转的仿射变换矩阵  
    float m[6];              
    CvMat M = cvMat( 2, 3, CV_32F, m );  
    cv2DRotationMatrix( center, degree,1, &M);  
    //变换图像，并用黑色填充其余值  
    cvWarpAffine(img,img_rotate, &M,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0) );  
}  

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
	string filename = argv[1];
	int width = image0->width;
	int height = image0->height;
	int nchannels = image0->nChannels;

	IplImage * distImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
			if(val > 70) 
			{
				CV_IMAGE_ELEM(distImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(distImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(distImg, unsigned char, j, 3*i+2) = 0;
			}
			else 
			{
				CV_IMAGE_ELEM(distImg, unsigned char, j, 3*i) = 255;
				CV_IMAGE_ELEM(distImg, unsigned char, j, 3*i+1) = 255;
				CV_IMAGE_ELEM(distImg, unsigned char, j, 3*i+2) = 255;
			}
		}
	}

	string filename1 = filename + ".bin.png";
	cvSaveImage(filename1.c_str(), distImg);

	 //IplConvKernel *element = cvCreateStructuringElementEx(5, 5, 0, 0, CV_SHAPE_ELLIPSE);
	 //cvMorphologyEx(distImg, distImg, NULL, element, CV_MOP_CLOSE);//关运算，填充内部的细线

	int max_color = 256*256*256 - 1;
	int color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(distImg, cvPoint(w,h), CV_RGB(hig, mid, low));
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

	vector<int>colorsum(color, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= color)
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
	for(int i = 0; i < color; i++)
		cout<<colorsum[i]<<" ";
	cout<<endl;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val > 0)
			{
				if (colorsum[val] > 70000)
				{  
					CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w) = 0;  
					CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+1) = 0;  
					CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+2) = 0;  
				}
				else
				{
					CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w) = 255;  
					CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+1) = 255;  
					CV_IMAGE_ELEM(distImg, unsigned char, h, 3*w+2) = 255;  
				}
			}
		}
	}

	string filename2 = filename + ".fill.png";
	cvSaveImage(filename2.c_str(), distImg);

	cvReleaseImage(&image0);
	cvReleaseImage(&distImg);
	return 0;

}
