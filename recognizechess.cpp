#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；

#include <iostream>
#include <string>

using namespace cv;
using namespace std;

#define DOUBLE_INF 1.79e+308

enum ChessType {BLACK_CHE = 0, BLACK_MA, BLACK_XIANG, BLACK_SHI, BLACK_JIANG, BLACK_ZU,
	RED_CHE, RED_MA, RED_XIANG, RED_SHI, RED_JIANG, RED_ZU, LAST_POS, UNKNOWNCHESS};

string chessnames[16] = {
	"BC", "BM", "BX", "BS", "BJ", "BP","BZ", 
	"RC", "RM", "RX", "RS", "RJ", "RP","RZ", 
	"LP", "NA"
};

IplImage ** templChesses = 0;

bool initTemplChesses()
{
	templChesses = new IplImage * [15];
	int id = 0;
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_che.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_ma.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_xiang.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_shi.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_jiang.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_pao.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_black_zu.png", 1);

	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_che.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_ma.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_xiang.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_shi.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_jiang.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_pao.png", 1);
	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_red_zu.png", 1);

	templChesses[id++] = cvLoadImage("/Users/xiaohang/Test/xiangqi2/sift_imgs/xiangqi_blank.png", 1);

	bool isok = true;
	for(int i = 0; i < 15; i++)
	{
		if(templChesses[i] == 0)
		{
			cout<<"unable to load "<<chessnames[i]<<endl;
			isok = false;
		}
	}
	if(!isok) return false;

	int width = templChesses[0]->width;
	int height = templChesses[0]->height;
	int hlfwid = width/2;
	int hlfhei = height/2;
	int nchannels = templChesses[0]->nChannels;
	for(int i = 0; i < 15; i++)
	{
		if(templChesses[i]->width != width ||
				templChesses[i]->height != height ||
				templChesses[i]->nChannels != nchannels)
		{
			cout<<"template images have different size"<<endl;
			return false;
		}
		//cvSmooth(templChesses[i], templChesses[i], CV_GAUSSIAN, 3,3,0,0);
		IplImage * hlfImg = cvCreateImage(cvSize(hlfwid, hlfhei), IPL_DEPTH_8U, nchannels);
		cvResize(templChesses[i], hlfImg);
		cvReleaseImage(&templChesses[i]);
		templChesses[i] = hlfImg;
	}
	return isok;
}

double bestMatchingScore(IplImage * smallImg, IplImage * bigImg)
{
	if(!smallImg || !bigImg) return DOUBLE_INF;
	int swid = smallImg->width;
	int shei = smallImg->height;
	int bwid = bigImg->width;
	int bhei = bigImg->height;
	if(swid > bwid || shei > bhei)
	{
		cout<<"smallImg should be smaller than bigImg"<<endl;
		return DOUBLE_INF;
	}
	if(smallImg->nChannels != bigImg->nChannels)
	{
		cout<<"smallImg and bigImg have different channels"<<endl;
		return DOUBLE_INF;
	}
	int nchannels = smallImg->nChannels;
	double min_sum = -1;
	for(int dy = 0; dy <= bhei-shei; dy++)
	{
		for(int dx = 0; dx <= bwid-swid; dx++)
		{
			double sum = 0;
			for(int sy = 0; sy < shei; sy++)
			{
				for(int sx = 0; sx < swid; sx++)
				{
					int by = sy + dy;
					int bx = sx + dx;
					if(bx < 0 || bx >= bwid) continue;
					if(by < 0 || by >= bhei) continue;

					for(int c = 0; c < nchannels; c++)
					{
						int sind = sx * nchannels + c + sy * smallImg->widthStep;
						int bind = bx * nchannels + c + by * bigImg->widthStep;
						int diff = (int)(bigImg->imageData[bind]) - (int)(smallImg->imageData[sind]);
						if(diff < 0 ) diff = -diff;
						sum += diff;
					}
				}
			}
			if(min_sum == -1 || sum < min_sum) min_sum = sum;
		}
	}
	return min_sum;
}

int whichChess(IplImage * bigImage)
{
	int width = bigImage->width;
	int height = bigImage->height;
	double min_score = -1;
	int min_id = -1;
	for(int i = 0; i < 15; i++)
	{
		double score = bestMatchingScore(templChesses[i], bigImage);
		if(min_score == -1)
		{
			min_score = score;
			min_id = i;
		}
		else if(score < min_score)
		{
			min_score = score;
			min_id = i;
		}
	}
	cout<<"min_score = "<<min_score<<" "<<chessnames[min_id]<<endl;
	return min_id;
}

IplImage * cropImage(IplImage * src, int x, int y, int width, int height)
{
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}

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
	string filename = argv[1];
	string filename2;
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

	filename2 = filename + ".grid0.png";
	cvSaveImage(filename2.c_str(), image0);

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
	
	int height2 = image1->height/(double)image1->width * 540.0 + 0.5;
	IplImage * boardImage = cvCreateImage(cvSize(540, height2), IPL_DEPTH_8U, image1->nChannels);
	cvResize(image1, boardImage);
	//
	//the font variable    
	CvFont font;    
	double hScale=1;   
	double vScale=1;    
	int lineWidth=2;// 相当于写字的线条    
	// 初始化字体   
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);//初始化字体，准备写到图片上的   

	initTemplChesses();
	for(int j = 0; j < 10; j++)
	{
		for(int i = 0; i < 9; i++)
		{
			int x = bst_x0 + i * bst_xstep;
			int y = bst_y0 + j * bst_ystep;
			IplImage * chessImg = cropImage(boardImage, x - 30, y - 30, 61, 61);

			IplImage * hlfChessImg = cvCreateImage(cvSize(30, 30), IPL_DEPTH_8U, chessImg->nChannels);
			cvResize(chessImg, hlfChessImg);
			cvReleaseImage(&chessImg);
			chessImg = hlfChessImg;

			int type = whichChess(chessImg);
			cvReleaseImage(&chessImg);

			//cout<<chessnames[type]<<" ";
			string showMsg = chessnames[type];
			showMsg = showMsg.substr(1,1);
			//if(type == 14) showMsg = "E";
			// cvPoint 为起笔的x，y坐标   
			if(type < 7)
				cvPutText(image1,showMsg.c_str(),cvPoint(x-20,y),&font,CV_RGB(0,0,255));//在图片中输出字符
			else if(type < 14)
				cvPutText(image1,showMsg.c_str(),cvPoint(x-20,y),&font,CV_RGB(255,0,0));//在图片中输出字符
		}
		cout<<endl;
	}

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

	filename2 = filename + ".recog.png";
	cvSaveImage(filename2.c_str(), image1);
	cvReleaseImage(&image0);
	cvReleaseImage(&image1);
	cvReleaseImage(&boardImage);

	for(int i = 0; i < 15; i++) cvReleaseImage(&templChesses[i]);
	delete [] templChesses; templChesses = 0;
	return 0;
}
