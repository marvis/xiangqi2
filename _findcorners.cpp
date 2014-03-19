#include "cv.h"
#include "highgui.h"
#include<iostream>
int main(int argc,char** argv)
{
	IplImage* pImg;
	IplImage* pHarrisImg;
	IplImage* grayImage;
	IplImage* dst8;
	double minVal=0.0, maxVal=0.0;
	double scale, shift;
	double min=0, max=255;
	if((pImg=cvLoadImage(argv[1],1))!=NULL)
	{
		cvNamedWindow("source",1);
		cvShowImage("source",pImg);
		pHarrisImg=cvCreateImage(cvGetSize(pImg),IPL_DEPTH_32F,1);
		//there we should define IPL_DEPTH_32F rather than IPL_DEPTH_8U
		grayImage=cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
		dst8=cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);//this is for the result image
		grayImage->origin=pImg->origin;  //there make sure the same  origin between grayImage and pImg

		cvCvtColor(pImg,grayImage,CV_BGR2GRAY);//cause harris need gray scale image,we should convert RGB 2 gray
		for(int j = 0; j < grayImage->height; j++)
		{
			for(int i = 0; i < grayImage->width; i++)
			{
				double val = CV_IMAGE_ELEM(grayImage, unsigned char, j, i);
				if(val > 70)
					CV_IMAGE_ELEM(grayImage, unsigned char, j, i) = 255;
				else 
					CV_IMAGE_ELEM(grayImage, unsigned char, j, i) = 0;

			}
		}

		int block_size=7;
		//do harris algorithm
		cvCornerHarris(grayImage,pHarrisImg,block_size,3,0.04);

		//convert scale so that we see the clear image
		cvMinMaxLoc(pHarrisImg,&minVal,&maxVal,NULL,NULL,0);

		std::cout<<minVal<<std::endl;
		std::cout<<maxVal<<std::endl;

		scale=(max-min)/(maxVal-minVal);
		shift=-minVal*scale+min;
		cvConvertScale(pHarrisImg,dst8,scale,shift);

		double thresh = 50;
		for( int j = 0; j < dst8->height ; j++ )  
		{ for( int i = 0; i < dst8->width; i++ )  
			{  
				if(CV_IMAGE_ELEM(dst8, unsigned char, j, i) > thresh )  
				{  
					//cvCircle(dst8, cvPoint(i, j), 5, cvScalar(255)); 
					CV_IMAGE_ELEM(dst8, unsigned char, j, i) = 255;
				}  
				else CV_IMAGE_ELEM(dst8, unsigned char, j, i) = 0;
			}   
		} 

		cvNamedWindow("Harris",1);
		cvShowImage("Harris",dst8);
		cvWaitKey(0);
		cvDestroyWindow("source");
		cvDestroyWindow("Harris");
		cvReleaseImage(&dst8);
		cvReleaseImage(&pHarrisImg);
		return 0;
	}
	return 1;
}

