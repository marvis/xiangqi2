#include <stdio.h>  
#include "cv.h"  
#include "highgui.h"  
#include <string>
using namespace std;

#define MAX_CORNERS 200  

int main(int argc, char ** argv)  
{  
	int cornersCount=MAX_CORNERS;//得到的角点数目  
	CvPoint2D32f corners[MAX_CORNERS];//输出角点集合  
	vector<double> xcorners, ycorners;
	IplImage *srcImage = 0,*grayImage = 0,*corners1 = 0,*corners2 = 0;  
	int i;  
	CvScalar color = CV_RGB(0,0,255);  
	string filename = argv[1];
	cvNamedWindow("image",1);  

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

	cvGoodFeaturesToTrack(grayImage,corners1,  
			corners2,corners,  
			&cornersCount,0.05,  
			50,//角点的最小距离是30  
			0,//整个图像  
			3,0,0.4);  
	//默认值  
	printf("num corners found: %d/n",cornersCount);  

	// 画圆圈
	for( int i = 0; i < results->total; i++ )  
	{
		float* p = ( float* )cvGetSeqElem( results, i );  
		//霍夫圆变换  
		CvPoint pt = cvPoint( cvRound( p[0] ), cvRound( p[1] ) );  
		/*cvCircle(  
				srcImage,  
				pt,  //确定圆心  
				cvRound( p[2] ),  //确定半径  
				CV_RGB( 0x0, 0x0, 0xff )  
				);  //画圆函数  
		*/
		/*cvCircle(  
				srcImage,  
				pt,  //确定圆心  
				cvRound(2),  //确定半径  
				CV_RGB( 0x0, 0xff, 0x0 ),
				2, CV_AA, 0
				);  //画圆函数  
		*/
		xcorners.push_back(pt.x);
		ycorners.push_back(pt.y);

	}

	//开始画出每个点  
	if (cornersCount>0)  
	{  
		for (i=0;i<cornersCount;i++)  
		{  
			//cvCircle(srcImage,cvPoint((int)(corners[i].x),(int)(corners[i].y)), 2,color,2,CV_AA,0);  
			xcorners.push_back(corners[i].x);
			ycorners.push_back(corners[i].y);
		}  
	} 

	// =============================================== //
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
	int lineWidth=1;// 相当于写字的线条    
	// 初始化字体   
	cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);//初始化字体，准备写到图片上的

	// kmeans clustering on x and y coordinates
	int ncorners = xcorners.size();
	CvMat * xlocs = cvCreateMat(ncorners, 1, CV_32FC1);
	CvMat * xclusters = cvCreateMat(ncorners, 1, CV_32SC1);
	int xcluster_count = 9;

	CvMat * ylocs = cvCreateMat(ncorners, 1, CV_32FC1);
	CvMat * yclusters = cvCreateMat(ncorners, 1, CV_32SC1);
	int ycluster_count = 10;
	
	for(int i = 0; i < ncorners; i++)
	{
		cvmSet(xlocs, i, 0, xcorners[i]);
		cvmSet(ylocs, i, 0, ycorners[i]);
	}
/*
	cvKMeans2( xlocs, xcluster_count, xclusters,
			   cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.5 ));
	for (int i=0;i<ncorners;i++)  
	{  
		int clusterId = xclusters->data.i[i];
		cout<<clusterId<<" ";
		CvScalar color = color_tab[clusterId];
		cvCircle(srcImage,cvPoint((int)(xcorners[i]),(int)(ycorners[i])),  
				2,color,2,CV_AA,0);  
		char showMsg[1] = {'0' + clusterId};
		cvPutText(srcImage, showMsg, cvPoint((int)(xcorners[i]),(int)(ycorners[i])), &font, CV_RGB(0,0,255));
	}  
	cout<<endl;
*/
	cvKMeans2( ylocs, ycluster_count, yclusters,
			   cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 0.5 ));
	for (int i=0;i<ncorners;i++)  
	{  
		int clusterId = yclusters->data.i[i];
		cout<<clusterId<<" ";
		CvScalar color = color_tab[clusterId];
		cvCircle(srcImage,cvPoint((int)(xcorners[i]),(int)(ycorners[i])),  
				2,color,2,CV_AA,0);  
		char showMsg[1] = {'0' + clusterId};
		cvPutText(srcImage, showMsg, cvPoint((int)(xcorners[i]),(int)(ycorners[i])), &font, CV_RGB(0,0,255));
	}

	cvShowImage("image",srcImage);  
	filename = filename + ".kmeans.png";
	cvSaveImage(filename.c_str(),srcImage);  
	cvWaitKey(0);  

	cvReleaseImage(&srcImage);  
	cvReleaseImage(&grayImage);  
	cvReleaseImage(&gaussImage);  
	cvReleaseImage(&corners1);  
	cvReleaseImage(&corners2);  

	cvReleaseMat(&xlocs);
	cvReleaseMat(&xclusters);
	cvReleaseMat(&ylocs);
	cvReleaseMat(&yclusters);

	cvClearSeq(results);
	cvReleaseMemStorage(&storage);
	return 0;  
} 
