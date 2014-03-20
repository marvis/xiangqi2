#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；

#include <iostream>
#include <string>

using namespace cv;
using namespace std;
vector<double> smoothData(vector<double> data, int radius)
{
	if(data.empty()) return vector<double>();
	if(radius < 1) return data;

	int n1 = data.size();
	int n2 = n1 + 2*radius;
	vector<double> fulldata(n2, 0);
	for(int i = 0; i < radius; i++) 
	{
		fulldata[i] = data[0];
		fulldata[n1+radius+i] = data[n1-1];
	}
	for(int i = 0; i < n1; i++)fulldata[i+radius] = data[i];

	vector<double> smoothdata(n1, 0);
	double winsize = 2*radius+1;
	for(int i = 0; i < n1; i++)
	{
		double sumval = 0.0;
		for(int r = -radius; r <= radius; r++)
		{
			sumval+=fulldata[radius + i + r];
		}
		smoothdata[i] = sumval/winsize;
	}
	return smoothdata;
}
int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("No input image\n");
		return -1;
	}
	IplImage * image0 = cvLoadImage(argv[1], 0); // load as gray image
	IplImage * image1 = cvLoadImage(argv[1], 1); // load as color image
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
	
	// 去掉两边的黑色边框
	for(int j = 0; j < height; j++)
	{
		int i = 0;
		int val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
		while(val <= 30)
		{
			CV_IMAGE_ELEM(image0, unsigned char, j, i) = 255;
			i++;
			val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
		}
		i = width-1;
		val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
		while(val <= 30)
		{
			CV_IMAGE_ELEM(image0, unsigned char, j, i) = 255;
			i--;
			val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
		}
	}

	filename1 = filename0 + ".grid0.png";
	cvSaveImage(filename1.c_str(), image0);

	vector<double> xdata(width, 0);
	vector<double> ydata(height, 0);
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(image0, unsigned char, j, i);
			xdata[i] += 255.0 - val;
			ydata[j] += 255.0 - val;
		}
	}
	//xdata = smoothData(xdata, 1); // 窗口半径
	//ydata = smoothData(ydata, 1);

	int bst_x0 = 0;
	double bst_xstep = 0;
	double max_xscore = 0;
	int min_xstep = width/10.0, max_xstep = width/8.0 + 0.5;
	
	for(double xstep = min_xstep; xstep <= max_xstep; xstep+=0.5)
	{
		//cout<<xstep<<" : "<<endl;
		for(int x0 = 0; x0 + 8*xstep < width; x0++)
		{
			double score = 0.0;
			for(int i = 0; i < 9; i++) score += xdata[(int)(x0 + i*xstep)];
			//cout<<"\t"<<x0<<" : "<<score<<endl;
			if(score > max_xscore)
			{
				max_xscore = score;
				bst_x0 = x0;
				bst_xstep = xstep;
			}
		}
	}
	
	int bst_y0 = 0;
	double bst_ystep = 0;
	double max_yscore = 0;
	int min_ystep = height/11.0, max_ystep = height/9.0 + 0.5;
	
	for(double ystep = min_ystep; ystep <= max_ystep; ystep+=0.5)
	{
		for(int y0 = 0; y0 + 9*ystep < height; y0++)
		{
			double score = 0.0;
			for(int i = 0; i < 10; i++) score += ydata[(int)(y0 + i*ystep)];
			if(score > max_yscore)
			{
				max_yscore = score;
				bst_y0 = y0;
				bst_ystep = ystep;
			}
		}
	}
	cout<<"bst_x0 = "<<bst_x0<<" bst_xstep = "<< bst_xstep<<endl;
	cout<<"bst_y0 = "<<bst_y0<<" bst_ystep = "<< bst_ystep<<endl;

	for(int i = 0; i < 9; i++)
	{
		int x1 = bst_x0 + i * bst_xstep;
		int y1 = bst_y0;
		int x2 = x1;
		int y2 = bst_y0 + 9*bst_ystep;
		cvLine(image1, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(0xff, 0x0, 0x0));
	}
	for(int j = 0; j < 10; j++)
	{
		int x1 = bst_x0;
		int y1 = bst_y0 + j * bst_ystep;
		int x2 = bst_x0 + 8 * bst_xstep;
		int y2 = y1;
		cvLine(image1, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(0x0, 0xff, 0x0));
	}

	filename1 = filename0 + ".grid.png";
	cvSaveImage(filename1.c_str(), image1);
	cvReleaseImage(&image0);
	cvReleaseImage(&image1);
	return 0;
}
