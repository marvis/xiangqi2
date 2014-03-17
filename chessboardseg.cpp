//compile: g++ -mmacosx-version-min=10.8 test.cpp $OPENCV_LIBS

#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；
#include <cmath>
#include <iostream>
#include <stack>

using namespace cv;
using namespace std;
#define DEBUG 0

IplImage * cropImage(IplImage * src, int x, int y, int width, int height)
{
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}

double bworient_angle(IplImage * binImg)
{
	double sum_x = 0, sum_y = 0;
	int width = binImg->width;
	int height = binImg->height;
	int npixels = 0;
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(CV_IMAGE_ELEM(binImg, unsigned char, h, w) > 0)
			{
				sum_x += w;
				sum_y += h;
				npixels++;
			}
		}
	}
	
	if(npixels == 0) return 0;
	double avg_x = sum_x/npixels;
	double avg_y = sum_y/npixels;
	
	double a = 0, b = 0, c = 0;
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(CV_IMAGE_ELEM(binImg, unsigned char, h, w) > 0)
			{
				a += (w - avg_x) * (w - avg_x);
				c += (h - avg_y) * (h - avg_y);
				b += 2 * (w-avg_x) * (h-avg_y);
			}
		}
	}
	if(a == c) return 3.1415926/2.0;
	else return 0.5 * atan(b/(a-c));
}

int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("No input image\n");
		return -1;
	}
	IplImage * image0 = cvLoadImage(argv[1], 1);
	if(!image0)
	{
		printf( "No image data \n" );
		return -1;
	}
	string filename = argv[1];
	int width = image0->width;
	int height = image0->height;
	int nchannels = image0->nChannels;

	double * distData = new double[width * height];
	double boardColor[3] = {38.0, 96.0, 211.0}; // BGR
	double max_dist = 0, min_dist = 1000000;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double sum = 0;
			int ind0 = j * image0->widthStep + i * nchannels;
			int ind1 = j*width + i;
			for(int c = 0; c < nchannels; c++)
			{
				double val0 = (unsigned char)(image0->imageData[ind0 + c]);
				if(val0 < 0)
				{
					val0 += 256;
					cout<<"("<<j<<","<<i<<","<<c<<") = "<< val0 << endl;
				}
				double val1 = boardColor[c];
				sum += (val0-val1)*(val0-val1);
			}
			double dist = sqrt(sum);
			distData[ind1] = dist;
			if(dist > max_dist) max_dist = dist;
			if(dist < min_dist) min_dist = dist;
		}
	}

	max_dist -= min_dist;
	
	IplImage * distImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	int widthStep = distImg->widthStep;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			int ind = j * width + i;
			double val = (distData[ind] - min_dist)/max_dist;
			if(val > 0.3) distImg->imageData[j*widthStep + i] = 0;
			else distImg->imageData[j*widthStep + i] = 255; 
		}
	}

#if DEBUG
	string filename0 = filename + ".dist0.png";
	cvSaveImage(filename0.c_str(), distImg);
#endif

	 IplConvKernel *element = cvCreateStructuringElementEx(10, 10, 0, 0, CV_SHAPE_ELLIPSE);
	 cvMorphologyEx(distImg, distImg, NULL, element, CV_MOP_CLOSE);//关运算，填充内部的细线

#if DEBUG
	string filename1 = filename + ".dist1.png";
	cvSaveImage(filename1.c_str(), distImg);
#endif

	int color = 254;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color > 0)
			{
				if(CV_IMAGE_ELEM(distImg, unsigned char, h, w)==255)
				{
					cvFloodFill(distImg, cvPoint(w,h), CV_RGB(color, color, color));
					color--;
				}
			}
			else
			{
				//cout<<"Too many connected areas. "<<endl;
				//return -1;
			}
		}
	}

#if DEBUG
	string filename2 = filename + ".dist2.png";
	cvSaveImage(filename2.c_str(), distImg);
#endif

	int colorsum[255] = {0};
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(CV_IMAGE_ELEM(distImg, unsigned char, h, w) > 0)
			{
				colorsum[CV_IMAGE_ELEM(distImg, unsigned char, h, w)]++; //统计每种颜色的数量
			}
		}
	}
	vector<int> v1(colorsum, colorsum+255);//用数组初始化vector
	//求出最多数量的染色，注意max_element的使用方法
	int maxcolorsum = max_element(v1.begin(), v1.end()) - v1.begin();
	printf("maxcolorsum = %d\n", maxcolorsum);

	int min_w = width, max_w = 0;
	int min_h = height, max_h = 0;
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if (CV_IMAGE_ELEM(distImg, unsigned char, h, w) == maxcolorsum)  
            {  
                CV_IMAGE_ELEM(distImg, unsigned char, h, w) = 255;  
				min_w = (w < min_w) ? w : min_w;
				max_w = (w > max_w) ? w : max_w;
				min_h = (h < min_h) ? h : min_h;
				max_h = (h > max_h) ? h : max_h;
            }  
            else  
            {  
                CV_IMAGE_ELEM(distImg, unsigned char, h, w) = 0;  
            }
		}
	}

#if DEBUG
	string filename3 = filename + ".dist3.png";
	cvSaveImage(filename3.c_str(), distImg);
#endif

#if DEBUG
	cvRectangle(distImg, cvPoint(min_w, min_h), cvPoint(max_w, max_h), cvScalar(255));
	cvRectangle(image0, cvPoint(min_w, min_h), cvPoint(max_w, max_h), cvScalar(255, 0, 0));
	string filename4 = filename + ".dist4.png";
	cvSaveImage(filename4.c_str(), distImg);
	string filename5 = filename + ".dist5.png";
	cvSaveImage(filename5.c_str(), image0);
#endif

	if(min_w >= 20) min_w = min_w - 20;
	if(max_w + 20 < width) max_w = max_w + 20;
	if(min_h >= 20) min_h = min_h - 20;
	if(max_h + 20 < height) max_h = max_h + 20;
	width = max_w - min_w + 1;
	height = max_h - min_h + 1;
	IplImage * boardImg = cropImage(distImg, min_w, min_h, width, height);
	
	double theta = bworient_angle(boardImg);	
	cout<<"theta = "<<theta * 180/3.1415926<<endl;

	string filename6 = filename + ".chess.png";
	cvSaveImage(filename6.c_str(), boardImg);

	cvReleaseImage(&image0);
	cvReleaseImage(&distImg);
	cvReleaseImage(&boardImg);
	delete [] distData; distData = 0;
	return 0;

}
