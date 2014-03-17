//compile: g++ -mmacosx-version-min=10.8 test.cpp $OPENCV_LIBS

#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("No input image\n");
		return -1;
	}
	IplImage * image = cvLoadImage(argv[1], 1);
	if(!image)
	{
		printf( "No image data \n" );
		return -1;
	}
	string filename = argv[1];
	IplImage * newImg8U = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int width = image->width;
	int height = image->height;
	int nchannels = image->nChannels;
	int widthStep = image->widthStep;

	unsigned short * data = new unsigned short[width*height];

	int max_val = 0;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			unsigned short sum = 0;
			for(int c = 0; c < nchannels; c++)
			{
				sum += image->imageData[i*nchannels + c + j * widthStep];
			}
			data[i+j*width] = sum;
			max_val = (sum > max_val) ? sum : max_val;
		}
	}

	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			newImg8U->imageData[i + j*newImg8U->widthStep] = (unsigned char)(data[i+j*width]*255.0/max_val);
		}
	}
	
	filename = filename + ".new.png";
	cvSaveImage(filename.c_str(), newImg8U);
	cvReleaseImage(&image);
	cvReleaseImage(&newImg8U);
	return 0;

}
