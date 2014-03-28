#include "opencv/cv.h"   //cv.h OpenCV的主要功能头文件，务必要；
#include "opencv/highgui.h" //显示图像用的，因为用到了显示图片，所以需要包含进去；
#include <cmath>
#include <iostream>
#include <stack>
#include <fstream>
#include "svm.h"
#include "classes/feature.h"

using namespace cv;
using namespace std;
#define DEBUG 0

enum ChessType {BLACK_CHE = 0, BLACK_MA, BLACK_XIANG, BLACK_SHI, BLACK_JIANG, BLACK_PAO, BLACK_ZU,
	           RED_CHE,        RED_MA,   RED_XIANG,   RED_SHI,   RED_JIANG,   RED_PAO,   RED_ZU, 
			   NOCHESS, UNKNOWNCHESS, CHE, MA, PAO};
int classtypes[12] = {NOCHESS, CHE, MA, BLACK_XIANG, RED_XIANG, BLACK_SHI, RED_SHI, BLACK_JIANG, RED_JIANG, RED_ZU, BLACK_ZU, PAO};
string chessnames[] = {
	"BC", "BM", "BX", "BS", "BJ", "BP","BZ", 
	"RC", "RM", "RX", "RS", "RJ", "RP","RZ", 
	"  ", "XX", " C", " M", " P"
};

string filename0, filename1;
string modelfile = "classes/trainfile_scale.model";
string rulefile = "classes/rules";

struct svm_model* model = 0;
vector<pair<double, double> > scale_params;
bool load_svm_rules(string rulefile)
{
	assert(scale_params.empty());
	ifstream ifs;
	ifs.open(rulefile.c_str());
	char name[256];
	ifs.getline(name, 256);
	ifs.getline(name, 256);
	int id;
	double min_val, max_val;
	while(ifs.good())
	{
		ifs >> id;
		ifs >> min_val;
		ifs >> max_val;
		scale_params.push_back(pair<double,double>(min_val, max_val));
	}
	ifs.close();
	return true;
}

void scale_svm_node(struct svm_node * x, vector<pair<double, double> > &scale_params, double scale_min_val, double scale_max_val)
{
	int nFeat = scale_params.size();
	for(int i = 0; i < nFeat; i++)
	{
		double min_val = scale_params[i].first;
		double max_val = scale_params[i].second;
		//x[i].index = i+1;
		x[i].value = (x[i].value - min_val)/(max_val - min_val) * 2.0 - 1.0;
		//cout<<" "<<x[i].index<<":"<<x[i].value;
	}
	//cout<<endl;
}

IplImage * cropImage(IplImage * src, int x, int y, int width, int height)
{
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}
IplImage * cropImage2(IplImage * src, int x, int y, int width, int height)
{
	if(x < 0) x = 0;
	if(y < 0) y = 0;
	if(x + width > src->width) x = src->width - width;
	if(y + height > src->height) y = src->height - height;
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}

CvRect getMaskBounding(IplImage * mask)
{
	CvRect rect;
	if(!mask || mask->depth != IPL_DEPTH_8U || mask->nChannels != 1)
	{
		cerr<<"Error: getMaskBounding - mask should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return rect;
	}

	int width = mask->width;
	int height = mask->height;
	int min_i = width - 1, max_i = 0;
	int min_j = height - 1, max_j = 0;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(CV_IMAGE_ELEM(mask, unsigned char, j, i) > 0)
			{
				min_i = (i < min_i) ? i : min_i;
				max_i = (i > max_i) ? i : max_i;
				min_j = (j < min_j) ? j : min_j;
				max_j = (j > max_j) ? j : max_j;
			}
		}
	}
	rect.x = min_i;
	rect.y = min_j;
	rect.width = max_i - min_i + 1;
	rect.height = max_j - min_j + 1;
	return rect;
}

IplImage * getMaximumComponent(IplImage * src, IplImage * mask, double threshold)
{
	if(!src || src->depth != IPL_DEPTH_8U)
	{
		cout<<"Invalid src image"<<endl;
		return 0;
	}
	if(mask)
	{
		if(src->width != mask->width || src->height != mask->height)
		{
			cout<<"Size error"<<endl;
			return 0;
		}
	}
	int width = src->width;
	int height = src->height;
	IplImage * binImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); // 由于floodfill不支持16位数据, 所以用3通道
	
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(src, unsigned char, j, i);
			int maskval = mask ? CV_IMAGE_ELEM(mask, unsigned char, j, i) : 1;
			if(maskval > 0 && val >= threshold)
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
			}
			else
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
			}
		}
	}

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
				return 0;
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
				return 0;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}

	int maxcolorLabel = max_element(colorsum.begin(), colorsum.end()) - colorsum.begin();

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
			}
			else
			{  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 0;  
			}
		}
	}
	return binImg;
}

IplImage * cropImageToRect(IplImage * src, CvPoint tlPoint, CvPoint trPoint, CvPoint blPoint, CvPoint brPoint)
{
	double dx1 =  (trPoint.x + brPoint.x)/2.0 - (tlPoint.x + blPoint.x)/2.0;
	double dy1 =  (trPoint.y + brPoint.y)/2.0 - (tlPoint.y + blPoint.y)/2.0;
	double dx2 = (tlPoint.x + trPoint.x)/2.0 - (blPoint.x + brPoint.x)/2.0;
	double dy2 = (tlPoint.y + trPoint.y)/2.0 - (blPoint.y + brPoint.y)/2.0;
	int width = sqrt(dx1*dx1 + dy1*dy1) + 0.5;
	int height = sqrt(dx2*dx2 + dy2*dy2) + 0.5;
	IplImage * dst = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, src->nChannels);
	int nchannels = src->nChannels;
	Point2f srcPoints[4];
	Point2f dstPoints[4];
	
	srcPoints[0].x = 0;           dstPoints[0].x = tlPoint.x;
	srcPoints[0].y = 0;           dstPoints[0].y = tlPoint.y;

	srcPoints[1].x = width -1;    dstPoints[1].x = trPoint.x;
	srcPoints[1].y = 0;           dstPoints[1].y = trPoint.y;

	srcPoints[2].x = 0;           dstPoints[2].x = blPoint.x;
	srcPoints[2].y = height - 1;  dstPoints[2].y = blPoint.y;

	srcPoints[3].x = width - 1;   dstPoints[3].x = brPoint.x;
	srcPoints[3].y = height - 1;  dstPoints[3].y = brPoint.y;

	Mat t = getPerspectiveTransform(srcPoints,dstPoints);
	//printf("transform matrix\n");  
    /*for(int i =0;i<3;i++)  
    {  
        printf("% .4f ",t.at<double>(0,i));  
        printf("% .4f ",t.at<double>(1,i));  
        printf("% .4f \n",t.at<double>(2,i));  
    }*/

	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			Mat sample = (Mat_<double>(3,1)<<i,j,1);
			Mat r = t*sample;
			double s = r.at<double>(2,0);
			int x = round(r.at<double>(0,0)/s);
			int y = round(r.at<double>(1,0)/s);

			for(int c = 0; c < nchannels; c++)
			{
				CV_IMAGE_ELEM(dst, unsigned char, j, nchannels*i + c) = CV_IMAGE_ELEM(src, unsigned char, y, nchannels*x+c);
			}
		}
	}
	return dst;
}

IplImage * cropImageToRect_False(IplImage * src, CvPoint tlPoint, CvPoint trPoint, CvPoint blPoint, CvPoint brPoint)
{
	double dx1 =  (trPoint.x + brPoint.x)/2.0 - (tlPoint.x + blPoint.x)/2.0;
	double dy1 =  (trPoint.y + brPoint.y)/2.0 - (tlPoint.y + blPoint.y)/2.0;
	double dx2 = (tlPoint.x + trPoint.x)/2.0 - (blPoint.x + brPoint.x)/2.0;
	double dy2 = (tlPoint.y + trPoint.y)/2.0 - (blPoint.y + brPoint.y)/2.0;
	int width = sqrt(dx1*dx1 + dy1*dy1) + 0.5;
	int height = sqrt(dx2*dx2 + dy2*dy2) + 0.5;
	IplImage * dst = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, src->nChannels);
	int nchannels = src->nChannels;
	for(int j = 0; j < height; j++)
	{
		double lambdaj = j/(height-1.0);
		for(int i = 0; i < width; i++)
		{
			double lambdai = i/(width-1.0);
			double lambda_tl = (1-lambdai)*(1-lambdaj);
			double lambda_tr = lambdai * (1-lambdaj);
			double lambda_bl = (1-lambdai)*lambdaj;
			double lambda_br = lambdai * lambdaj;
			int x = lambda_tl * tlPoint.x + lambda_tr * trPoint.x + lambda_bl * blPoint.x + lambda_br * brPoint.x + 0.5;
			int y = lambda_tl * tlPoint.y + lambda_tr * trPoint.y + lambda_bl * blPoint.y + lambda_br * brPoint.y + 0.5;
			for(int c = 0; c < nchannels; c++)
			{
				CV_IMAGE_ELEM(dst, unsigned char, j, nchannels*i + c) = CV_IMAGE_ELEM(src, unsigned char, y, nchannels*x+c);
			}
		}
	}
	return dst;
}

CvPoint crossPoint(float p1[4], float p2[4])
{
	// (y - y0)/vy0 = (x - x0)/vx0
	// vy0 * x - vy0 * x0 = vx0*y - vx0 * y0
	// vy0 * x + (-vx0)*y + vx0*y0 - vy0*x0 = 0
	double vx0 = p1[0];
	double vy0 = p1[1];
	double x0 = p1[2];
	double y0 = p1[3];
	double vx1 = p2[0];
	double vy1 = p2[1];
	double x1 = p2[2];
	double y1 = p2[3];
	double a0 = vy0;
	double b0 = -vx0;
	double c0 = vx0*y0 - vy0*x0;
	double a1 = vy1;
	double b1 = -vx1;
	double c1 = vx1*y1 - vy1*x1;
	CvPoint p;
	p.x = (b0*c1 - b1*c0)/(b1*a0 - b0*a1);
	p.y = (a0*c1 - a1*c0)/(a1*b0 - a0*b1);
	return p;
}

// image1 is the the original image with bgr channels
int xiaomiScreen(IplImage * image1, IplImage * screenMask)
{
	assert(image1->width == screenMask->width && image1->height == screenMask->height);
	int width = image1->width;
	int height = image1->height;
	IplImage * image0 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	IplImage * binImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3); // to store the filled color
	cvCvtColor(image1, image0, CV_BGR2GRAY);
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

#if DEBUG
	filename1 = filename0 + ".bin1.png";
	cvSaveImage(filename1.c_str(), binImg);
#endif

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

#if DEBUG
	filename1 = filename0 + ".bin2.png";
	cvSaveImage(filename1.c_str(), binImg);
#endif

	IplConvKernel *element = cvCreateStructuringElementEx(10, 10, 0, 0, CV_SHAPE_ELLIPSE);
	cvMorphologyEx(binImg, binImg, NULL, element, CV_MOP_CLOSE);//关运算，填充内部的细线

#if DEBUG
	filename1 = filename0 + ".bin3.png";
	cvSaveImage(filename1.c_str(), binImg);
#endif
/*
	IplImage * binImg2 = cropImage(binImg, min_w, min_h, (max_w-min_w+1), (max_h-min_h+1));
	cvReleaseImage(&binImg);
	
	binImg = binImg2;
	width = binImg->width;
	height = binImg->height;
*/
	// find maximum component second time
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(i < min_w || i > max_w || j < min_h || j > max_h)
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
			}
			else
			{
				unsigned char b = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i);
				unsigned char g = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1);
				unsigned char r = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2);

				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255 - b;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255 - g;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255 - r;
			}
		}
	}

#if DEBUG
	filename1 = filename0 + ".bin4.png";
	cvSaveImage(filename1.c_str(), binImg);
#endif

	max_color = 256*256*256 - 1;
	color = 1;

	for(int h = min_h; h <= max_h; h++)
	{
		for(int w = min_w; w <= max_w; w++)
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
	colorNum = color;

	//colorsum.resize(colorNum, 0);
	vector<int> colorsum2(colorNum, 0);
	for(int h = min_h; h <= max_h; h++)
	{
		for(int w = min_w; w <= max_w; w++)
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
				colorsum2[rgb]++; //统计每种颜色的数量
			}
		}
	}

	maxcolorLabel = max_element(colorsum2.begin(), colorsum2.end()) - colorsum2.begin();
	//printf("maxcolorLabel = %d\n", maxcolorLabel);

	for(int h = min_h; h <= max_h; h++)
	{
		for(int w = min_w; w <= max_w; w++)
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
			}
			else
			{  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 0;  
			}
		}
	}

#if DEBUG
	filename1 = filename0 + ".bin5.png";
	cvSaveImage(filename1.c_str(), binImg);
#endif
	cvCvtColor(binImg, screenMask, CV_BGR2GRAY);

	cvReleaseImage(&image0);
	cvReleaseImage(&binImg);

	return true;
}

void getFourCorners(IplImage * mask, CvPoint & tlPoint, CvPoint & trPoint, CvPoint & blPoint, CvPoint & brPoint)
{
	CvRect rect = getMaskBounding(mask);

	// 找4个顶点的方法

	double mid_w = rect.x + rect.width/2.0;//(mtlx+mtrx+mblx+mbrx)/4.0;
	double mid_h = rect.y + rect.height/2.0;//(mtly+mtry+mbly+mbry)/4.0;
	double max_tl_dist = 0;
	double max_tr_dist = 0;
	double max_bl_dist = 0;
	double max_br_dist = 0;
	int Mtlx, Mtly;
	int Mtrx, Mtry;
	int Mblx, Mbly;
	int Mbrx, Mbry;

	for(int h = rect.y; h < rect.y + rect.height; h++)
	{
		for(int w = rect.x; w < rect.x + rect.width; w++)
		{
			int val = CV_IMAGE_ELEM(mask, unsigned char, h, w);
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
	
	tlPoint.x = Mtlx; tlPoint.y = Mtly;
	trPoint.x = Mtrx; trPoint.y = Mtry;
	blPoint.x = Mblx; blPoint.y = Mbly;
	brPoint.x = Mbrx; brPoint.y = Mbry;
}

// image0 is a grayscale image
// note: the boundary of image0 will be filled with white
void findgrids(IplImage * image0, int &bst_x0, int &bst_y0, double &bst_xstep, double & bst_ystep)
{
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

#if DEBUG
	filename1 = filename0 + ".grid0.png";
	cvSaveImage(filename1.c_str(), image0);
#endif

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

	bst_x0 = 0;
	bst_xstep = 0;
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
	
	bst_y0 = 0;
	bst_ystep = 0;
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
}

// return the maximum connected component of src into dst
bool maximumConnectedComponent(IplImage * src, IplImage * dst, double threshold, int max_value=255, int threshold_type = CV_THRESH_BINARY)
{
	if(!src || !dst || src->depth != IPL_DEPTH_8U ||dst->depth != IPL_DEPTH_8U)
	{
		cerr<<"Error: maximumConnectedComponent - src and dst should be non empty with type IPL_DEPTH_8U"<<endl;
		return false;
	}
	if(src->width != dst->width || src->height != dst->height)
	{
		cerr<<"Error: maximumConnectedComponent - src and dst should be the same size"<<endl;
		return false;
	}
	int width = src->width;
	int height = src->height;
	IplImage * binImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); // 由于floodfill不支持16位数据, 所以用3通道
	
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(src, unsigned char, j, i);
			if(threshold_type == CV_THRESH_BINARY)
			{
				if(val > threshold)
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
				}
				else
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
				}
			}
			else if(threshold_type == CV_THRESH_BINARY_INV)
			{
				if(val < threshold)
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
				}
				else
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
				}
			}
		}
	}

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
				cerr<<"Error: maximumConnectedComponent - too many connected areas. "<<endl;
				return false;
			}
		}
	}
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
				cerr<<"Error: maximumConnectedComponent - invalid rgb"<<endl;
				return false;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}

	int maxcolorLabel = max_element(colorsum.begin(), colorsum.end()) - colorsum.begin();

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val == maxcolorLabel)
				CV_IMAGE_ELEM(dst, unsigned char, h, w) = max_value;  
			else
				CV_IMAGE_ELEM(dst, unsigned char, h, w) = 0;  
		}
	}
	cvReleaseImage(&binImg);
	return true;
}

int whichChess(IplImage * src, string prefix, double & avgr)
{
	int width = src->width;
	int height = src->height;
	assert(src->nChannels == 3);
	int chessType = UNKNOWNCHESS;

	IplImage * src0 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	cvCvtColor(src, src0, CV_BGR2GRAY);
	IplImage * mask0 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); // OTSU threshold
	IplImage * mask1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); // maximum connected component
	IplImage * mask2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1); // fill holes
	cvThreshold(src0, mask0, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	maximumConnectedComponent(mask0, mask1, 128, 255, CV_THRESH_BINARY);

	cvSetZero(mask2);

	// fill the mask 
	double sum = 0.0; // 
	int count = 0;
	vector<int> j1s(width), j2s(width);
	for(int i = 0; i < width; i++)
	{
		int j1 = 0, j2 = height-1;
		while(j1 < height && CV_IMAGE_ELEM(mask1, unsigned char, j1, i) == 0) j1++;
		while(j2 >= 0 && CV_IMAGE_ELEM(mask1, unsigned char, j2, i) == 0) j2--;
		j1s[i] = j1;
		j2s[i] = j2;
		if(j1 < j2)
		{
			for(int j = j1; j <= j2; j++)
			{
				if(CV_IMAGE_ELEM(mask1, unsigned char, j, i) == 0)// && CV_IMAGE_ELEM(src0, unsigned char, j, i) < 120)
				{
					sum += CV_IMAGE_ELEM(src0, unsigned char, j, i);
					count++;
				}
			}
		}
	}
	double avg = (count > 0) ? sum/count : 0;
	double sumr = 0.0; // sumr will sum the red channels
	count = 0;
	for(int i = 0; i < width; i++)
	{
		int j1 = j1s[i];
		int j2 = j2s[i];
		if(j1 < j2)
		{
			for(int j = j1; j <= j2; j++)
			{
				if(CV_IMAGE_ELEM(mask1, unsigned char, j, i) == 0 && CV_IMAGE_ELEM(src0, unsigned char, j, i) < avg)
				{
					CV_IMAGE_ELEM(mask2, unsigned char, j, i) = 255;
					sumr += CV_IMAGE_ELEM(src, unsigned char, j, 3*i + 2);
					count++;
				}
			}
		}
	}
	avgr = (count > 0) ? sumr/count: 0;
	//cout<<"avgr = "<<avgr;

	IplConvKernel *element = cvCreateStructuringElementEx(2, 2, 0, 0, CV_SHAPE_ELLIPSE);
	cvMorphologyEx(mask2, mask2, NULL, element, CV_MOP_OPEN);//关运算，填充内部的细线

	count = 0;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(CV_IMAGE_ELEM(mask2, unsigned char, j, i) == 255) count++;
		}
	}
	if(count == 0) chessType = NOCHESS;
	else
	{
		CvRect rect = getMaskBounding(mask2);
		if(rect.height < 25 || rect.width < 25) chessType = NOCHESS;
		else
		{
			double mid_x = rect.x + rect.width/2.0;
			double mid_y = rect.y + rect.height/2.0;
			rect.width = 56;// (rect.width > rect.height) ? rect.width : rect.height;
			rect.height = 56;//rect.width;
			rect.x = mid_x - rect.width/2.0;
			rect.y = mid_y - rect.height/2.0;
			IplImage * midImg = cropImage2(src0, rect.x, rect.y, rect.width, rect.height);
			//cvSaveImage(prefix.c_str(), midImg);
			IplImage * midImgBGR = cvCreateImage(cvGetSize(midImg), IPL_DEPTH_8U, 3);
			cvCvtColor(midImg, midImgBGR, CV_GRAY2BGR);

			FeatureMap ** map = new (FeatureMap*);
			getFeature(midImgBGR, 8, map);
			normalizehog(*map, 0.2);
			PCAFeature(*map);

			int nFeat = (*map)->numFeatures * (*map)->sizeX * (*map)->sizeY;
			float * feats = (*map)->map;
			struct svm_node * x = (struct svm_node*) malloc((nFeat+1) * sizeof(struct svm_node));
			for(int i = 0; i < nFeat; i++)
			{
				x[i].index = i+1;
				x[i].value = feats[i];
			}
			x[nFeat].index = -1;
			scale_svm_node(x, scale_params, -1.0, 1.0);
			double predict_label = svm_predict(model, x);
			chessType = classtypes[(int)predict_label];
			//if(chessType == CHE || chessType == MA || chessType == PAO) cout<<avgr;
			//if(chessType == CHE) chessType = (avgr > 150.0) ? RED_CHE : BLACK_CHE;
			//else if(chessType == MA) chessType = (avgr > 150.0) ? RED_MA : BLACK_MA;
			//else if(chessType == PAO) chessType = (avgr > 150.0) ? RED_PAO : BLACK_PAO;
			delete map;
			delete [] (*map);

			cvReleaseImage(&midImg);
			cvReleaseImage(&midImgBGR);
		}
	}
	cvReleaseImage(&src0);
	cvReleaseImage(&mask0);
	cvReleaseImage(&mask1);
	cvReleaseImage(&mask2);
	return chessType;
}

int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("No input image\n");
		return -1;
	}
	if((model = svm_load_model(modelfile.c_str())) == 0)
	{
		cerr<<"Can't open model file "<<modelfile<<endl;
		return 0;
	}
	if(!load_svm_rules(rulefile.c_str()))
	{
		cerr<<"Can't load svm rule file"<<rulefile<<endl;
		return 0;
	}
	IplImage * image1 = cvLoadImage(argv[1], 1); // load as color image
	IplImage * drawImg = cvLoadImage(argv[1], 1); // load as color image
	if(!image1 || !drawImg)
	{
		printf( "No image data \n" );
		return -1;
	}
	filename0 = argv[1];
	int width = image1->width;
	int height = image1->height;

	// find maximum component first time
	IplImage * screenMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	xiaomiScreen(image1, screenMask);

	CvPoint tlPoint, trPoint, blPoint, brPoint; 
	getFourCorners(screenMask, tlPoint, trPoint, blPoint, brPoint);

	CvScalar color1 =  CV_RGB( 0xff, 0x0, 0x0 );
	CvScalar color2 =  CV_RGB( 0x0, 0xff, 0xff );
	cvCircle(drawImg, tlPoint , cvRound(15), color2, 2, CV_AA, 0);  //画圆函数  
	cvCircle(drawImg, trPoint , cvRound(15), color2, 2, CV_AA, 0);  //画圆函数  
	cvCircle(drawImg, blPoint , cvRound(15), color2, 2, CV_AA, 0);  //画圆函数  
	cvCircle(drawImg, brPoint , cvRound(15), color2, 2, CV_AA, 0);  //画圆函数  

#if DEBUG
	filename1 = filename0 + ".bin7.png";
	cvSaveImage(filename1.c_str(), drawImg);
#endif

	IplImage * screenImg1 = cropImageToRect(image1, tlPoint, trPoint, blPoint, brPoint);
	IplImage * screenImg2 = cvCreateImage(cvSize(540, 960), IPL_DEPTH_8U, screenImg1->nChannels);
	cvResize(screenImg1, screenImg2);

	IplImage * chessboardImg1 = cropImage(screenImg2, 0, 0.156*screenImg2->height, screenImg2->width, 0.7*screenImg2->height);
	IplImage * chessboardImg0 = cvCreateImage(cvGetSize(chessboardImg1), IPL_DEPTH_8U, 1);
	cvCvtColor(chessboardImg1, chessboardImg0, CV_BGR2GRAY);

	filename1 = filename0 + ".screen.png";
	cvSaveImage(filename1.c_str(), screenImg2);

	filename1 = filename0 + ".chessboard.png";
	cvSaveImage(filename1.c_str(), chessboardImg1);

	int bst_x0, bst_y0;
	double bst_xstep, bst_ystep;
	findgrids(chessboardImg0, bst_x0, bst_y0, bst_xstep, bst_ystep);
	
	int matrix[10][9];
	double avgrs[10][9];
	double max_black_avgr = 0;
	double min_red_avgr = 256;
	for(int j = 0; j < 10; j++)
	{
		for(int i = 0; i < 9; i++)
		{
			int x = bst_x0 + i * bst_xstep;
			int y = bst_y0 + j * bst_ystep;
			IplImage * chessImg = cropImage2(chessboardImg1, x - 35, y - 35, 70, 70);
			ostringstream oss;
			oss << filename0<<"."<<j<<"."<<i<<".png";
			double avgr;
			int chessType = whichChess(chessImg, oss.str(), avgr);
			matrix[j][i] = chessType;
			avgrs[j][i] = avgr;
			if(chessType >= 0 && chessType <= 6) max_black_avgr = (avgr > max_black_avgr) ? avgr : max_black_avgr;
			else if(chessType >= 7 && chessType <= 13) min_red_avgr = (avgr < min_red_avgr) ? avgr : min_red_avgr;
			cvReleaseImage(&chessImg);
			//cout<<" "<<chessnames[id]<<"  "<<endl;;
		}
		//cout<<endl;
	}
	double avgr_threshold = (max_black_avgr + min_red_avgr)/2.0;
	cout<<"avgr_threshold = "<<avgr_threshold<<endl;
	for(int j = 0; j < 10; j++)
	{
		for(int i = 0; i < 9; i++)
		{
			int chessType = matrix[j][i];
			double avgr = avgrs[j][i];
			if(chessType == CHE) matrix[j][i] = (avgr > avgr_threshold) ? RED_CHE : BLACK_CHE;
			if(chessType == MA) matrix[j][i] = (avgr > avgr_threshold) ? RED_MA : BLACK_MA;
			if(chessType == PAO) matrix[j][i] = (avgr > avgr_threshold) ? RED_PAO : BLACK_PAO;
			chessType = matrix[j][i];
			cout<<chessnames[chessType]<<"  ";
		}
		cout<<endl;
	}

	for(int i = 0; i < 9; i++)
	{
		int x1 = bst_x0 + i * bst_xstep;
		int y1 = bst_y0;
		int x2 = x1;
		int y2 = bst_y0 + 9*bst_ystep;
		cvLine(chessboardImg1, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(0xff, 0x0, 0x0));
	}
	for(int j = 0; j < 10; j++)
	{
		int x1 = bst_x0;
		int y1 = bst_y0 + j * bst_ystep;
		int x2 = bst_x0 + 8 * bst_xstep;
		int y2 = y1;
		cvLine(chessboardImg1, cvPoint(x1, y1), cvPoint(x2, y2), CV_RGB(0x0, 0xff, 0x0));
	}	

	filename1 = filename0 + ".grid.png";
	cvSaveImage(filename1.c_str(), chessboardImg1);

	cvReleaseImage(&image1);
	cvReleaseImage(&screenMask);
	cvReleaseImage(&screenImg2);
	cvReleaseImage(&chessboardImg1);
	cvReleaseImage(&chessboardImg0);
	cvReleaseImage(&drawImg);
	cvReleaseImage(&screenImg1);

	svm_free_model_content(model);
	svm_free_and_destroy_model(&model);
	return 0;

}
