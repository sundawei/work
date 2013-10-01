// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include "ConsoleApplication1.h"
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// The one and only application object

CWinApp theApp;

using namespace std;

using namespace cv;

/*

static void help()
{
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
            "Using the SURF desriptor:\n"
            "\n"
            "Usage:\n matcher_simple <image1> <image2>\n");
}

int find_match(Mat img_1,Mat img_2)
{
//			Mat img_1 = imread("c:\\1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//			Mat img_2 = imread("c:\\2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
			if(img_1.empty() || img_2.empty())
			{
				printf("Can't read one of the images\n");
				return -1;
			}

			int minHessian = 400;

			SurfFeatureDetector detector( minHessian );

			std::vector<KeyPoint> keypoints_1, keypoints_2;

			detector.detect( img_1, keypoints_1 );
			detector.detect( img_2, keypoints_2 );

			//-- Step 2: Calculate descriptors (feature vectors)
			SurfDescriptorExtractor extractor;

			Mat descriptors_1, descriptors_2;

			extractor.compute( img_1, keypoints_1, descriptors_1 );
			extractor.compute( img_2, keypoints_2, descriptors_2 );

			//-- Step 3: Matching descriptor vectors using FLANN matcher
			//-- Step 3: Matching descriptor vectors using FLANN matcher
			FlannBasedMatcher matcher;
			std::vector< DMatch > matches;
			matcher.match( descriptors_1, descriptors_2, matches );

			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for( int i = 0; i < descriptors_1.rows; i++ )
			{ double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
			}

			printf("-- Max dist : %f \n", max_dist );
			printf("-- Min dist : %f \n", min_dist );

			//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
			//-- PS.- radiusMatch can also be used here.
			std::vector< DMatch > good_matches;

			for( int i = 0; i < descriptors_1.rows; i++ )
			{ if( matches[i].distance < 2*min_dist )
			{ good_matches.push_back( matches[i]); }
			}  

			//-- Draw only "good" matches
			Mat img_matches;
			drawMatches( img_1, keypoints_1, img_2, keypoints_2, 
				good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), 
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ); 

			//-- Show detected matches
			imshow( "Good Matches", img_matches );
			cvWaitKey(0);
}

void cropImage(IplImage* src,IplImage* & dstimg,CvRect r)
{
	dstimg = cvCreateImage(cvSize(r.width,r.height),src->depth,src->nChannels);
	(((Mat)(src))(r)).convertTo( ((Mat)(dstimg)), ((Mat&)(dstimg)).type(),1,0);
}
void doma(Mat im1,Mat im2);
IplImage* src = 0; 
IplImage* dst = 0; 
void on_mouse( int event, int x, int y, int flags, void* ustc)
{
	static CvPoint pre_pt = {-1,-1};
	static CvPoint cur_pt = {-1,-1};
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	char temp[16];
	
	if( event == CV_EVENT_LBUTTONDOWN )
	{
		cvCopy(dst,src);
		sprintf(temp,"(%d,%d)",x,y);
		pre_pt = cvPoint(x,y);
		//cvPutText(src,temp, pre_pt, &font, cvScalar(0,0, 0, 255));
		//cvCircle( src, pre_pt, 3,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
		cvReleaseImage(&src);
		src=cvLoadImage("c:/1.jpg");
		cvShowImage( "src", src );
		cvCopy(src,dst);
	}
	else if( event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))
	{
		cvCopy(dst,src);
		sprintf(temp,"(%d,%d)",x,y);
		cur_pt = cvPoint(x,y);		
		//cvPutText(src,temp, cur_pt, &font, cvScalar(0,0, 0, 255));
		cvShowImage( "src", src );
	}
	else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		cvCopy(dst,src);
		sprintf(temp,"(%d,%d)",x,y);
		cur_pt = cvPoint(x,y);		
		//cvPutText(src,temp, cur_pt, &font, cvScalar(0,0, 0, 255));
		cvRectangle(src, pre_pt, cur_pt, cvScalar(0,255,0,0), 1, 8, 0 );
		cvShowImage( "src", src );
	}
	else if( event == CV_EVENT_LBUTTONUP )
	{
		if(abs(cur_pt.x-pre_pt.x)>8&&abs(cur_pt.y-pre_pt.y)>8)
		{
		sprintf(temp,"(%d,%d)",x,y);
		cur_pt = cvPoint(x,y);		
		//cvPutText(src,temp, cur_pt, &font, cvScalar(0,0, 0, 255));
		//cvCircle( src, cur_pt, 3,cvScalar(255,0,0,0) ,CV_FILLED, CV_AA, 0 );
		cvRectangle( src, pre_pt, cur_pt, cvScalar(0,255,0,0), 1, 8, 0 );
		cvShowImage( "src", src );
		cvCopy(src,dst);
		IplImage* cpi=NULL;
		cropImage(src,cpi,cvRect(pre_pt.x,pre_pt.y,abs(cur_pt.x-pre_pt.x),abs(cur_pt.y-pre_pt.y)));

		IplImage* cpigray=cvCreateImage(cvGetSize(cpi),IPL_DEPTH_8U,1);
		cvCvtColor(cpi,cpigray,CV_RGB2GRAY);

		//Mat i1 = cpi;
		//Mat i2 = imread("c:\\2.jpg");
		Mat i1=cpigray;
		Mat i2=imread("c:\\2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		//find_match(i1,i2);
		doma(i1,i2);

	//	cvShowImage("zzz",cpi);
		//cvWaitKey(0);
		cvReleaseImage(&cpigray);
		cvReleaseImage(&cpi);
		}

	}
}

/// 全局变量
Mat img; Mat templ; Mat result;
char* image_window = "Source Image";
char* result_window = "Result window";

int match_method;
int max_Trackbar = 5;

/// 函数声明
void MatchingMethod( int, void* );

void doma(Mat im1,Mat im2)
{
	  /// 载入原图像和模板块
  img = im2;
  templ = im1;//imread( argv[2], 1 );

  /// 创建窗口
  namedWindow( image_window, CV_WINDOW_NORMAL );
 // namedWindow( result_window, CV_WINDOW_AUTOSIZE );

  /// 创建滑动条
  char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
  createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );

  MatchingMethod( 0, 0 );

  waitKey(0);
}



int main()
{
	
	src=cvLoadImage("c:/1.jpg",1);
	dst=cvCloneImage(src);
	cvNamedWindow("src",CV_WINDOW_NORMAL);
	cvSetMouseCallback( "src", on_mouse, 0 );
	
	cvShowImage("src",src);
	cvWaitKey(0); 
	cvDestroyAllWindows();
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	return 0;
	
}




void MatchingMethod( int, void* )
{
  /// 将被显示的原图像
	Mat img_display=imread("c:/2.jpg");;
  //img.copyTo( img_display );

  /// 创建输出结果的矩阵
  int result_cols =  img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;

  result.create( result_cols, result_rows, CV_32FC1 );

  /// 进行匹配和标准化
  matchTemplate( img, templ, result, match_method );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

  /// 通过函数 minMaxLoc 定位最匹配的位置
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

  /// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

  /// 让我看看您的最终结果
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), cvScalar(0,0,255), 2, 8, 0 );
 // rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

  imshow( image_window, img_display );
 // imshow( result_window, result );

  return;
}

/*
int _tmain(int argc, TCHAR* argv[], TCHAR* envp[])
{
	int nRetCode = 0;

	HMODULE hModule = ::GetModuleHandle(NULL);

	if (hModule != NULL)
	{
		// initialize MFC and print and error on failure
		if (!AfxWinInit(hModule, NULL, ::GetCommandLine(), 0))
		{
			// TODO: change error code to suit your needs
			_tprintf(_T("Fatal Error: MFC initialization failed\n"));
			nRetCode = 1;
		}
		else
		{
			// TODO: code your application's behavior here.

		}
	}
	else
	{
		// TODO: change error code to suit your needs
		_tprintf(_T("Fatal Error: GetModuleHandle failed\n"));
		nRetCode = 1;
	}





	return nRetCode;
}
//*/

int main()
{	
	IplImage * src=cvLoadImage("c:/touying.jpg",0);
//	cvSmooth(src,src,CV_BLUR,3,3,0,0);
	cvThreshold(src,src,50,255,CV_THRESH_BINARY);
	IplImage* paintx=cvCreateImage( cvGetSize(src),IPL_DEPTH_8U, 1 );
	IplImage* painty=cvCreateImage( cvGetSize(src),IPL_DEPTH_8U, 1 );
	cvZero(paintx);
	cvZero(painty);
	int* v=new int[src->width*4];
	int* h=new int[src->height*4];
	memset(v,0,src->width*4);
	memset(h,0,src->height*4);
	
	int x,y;
	CvScalar s,t;
	for(x=0;x<src->width;x++)
	{
		for(y=0;y<src->height;y++)
		{
			s=cvGet2D(src,y,x);			
			if(s.val[0]==0)
				v[x]++;					
		}		
	}
	
	for(x=0;x<src->width;x++)
	{
		for(y=0;y<v[x];y++)
		{		
			t.val[0]=255;
			cvSet2D(paintx,y,x,t);		
		}		
	}
	
	for(y=0;y<src->height;y++)
	{
		for(x=0;x<src->width;x++)
		{
			s=cvGet2D(src,y,x);			
			if(s.val[0]==0)
				h[y]++;		
		}	
	}
	for(y=0;y<src->height;y++)
	{
		for(x=0;x<h[y];x++)
		{			
			t.val[0]=255;
			cvSet2D(painty,y,x,t);			
		}		
	}

	for(int y=1;y<src->height-1;y++)
	{
		if(h[y]==0)
		{
			if(h[y-1]!=0)
			for(x=0;x<src->width;x++)
			{
				cvSet2D(src,y,x,cvScalar(0,0,0));
			}
		}
		/*
		if(h[y]!=0)
		{
			if(h[y+1]==0)
			for(x=0;x<src->width;x++)
			{
				cvSet2D(src,y,x,cvScalar(0,0,0));
			}
		}
		//*/
	}



	cvNamedWindow("二值图像",1);
	cvNamedWindow("垂直积分投影",1);
	cvNamedWindow("水平积分投影",1);
	cvShowImage("二值图像",src);
	cvShowImage("垂直积分投影",paintx);
	cvShowImage("水平积分投影",painty);
	cvWaitKey(0);
	cvDestroyAllWindows();
	cvReleaseImage(&src);
	cvReleaseImage(&paintx);
	cvReleaseImage(&painty);
	return 0;
}