/*
  Detects SIFT features in two images and finds matches between them.

  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  @version 1.1.2-20100521
*/
#include <windows.h>
#include "sift.h"
#include "imgfeatures.h"
#include "kdtree.h"
#include "utils.h"
#include "xform.h"
#include "time.h"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN 最近点与次最近点距离之比*/
#define NN_SQ_DIST_RATIO_THR 0.49

int display = 0;

int main( int argc, char** argv )
{
	clock_t start22, end22;

	start22 = clock();


  IplImage* img1, * img2, * stacked;
  struct feature* feat1, * feat2, * feat;
  struct feature** nbrs;
  struct kd_node* kd_root;
  CvPoint pt1, pt2;
  double d0, d1;
  int n1, n2, k, i, m = 0;

  /*
  if( argc != 3 )
    fatal_error( "usage: %s <img1> <img2>", argv[0] );
  */
  
  //argv[1] = "beaver.png";
  //argv[2] = "beaver_xform.png";
  
  argv[1] = "DSC1_512.JPG";
  argv[2] = "DSC2_512.JPG";

  img1 = cvLoadImage( argv[1], 1 );
  if( ! img1 )
    fatal_error( "unable to load image from %s", argv[1] );
  img2 = cvLoadImage( argv[2], 1 );
  if( ! img2 )
    fatal_error( "unable to load image from %s", argv[2] );
  stacked = stack_imgs( img1, img2 );

  fprintf( stderr, "Finding features in %s...\n", argv[1] );
  n1 = sift_features( img1, &feat1 );	//计算图像特征点
    if( display )	//显示？
    {
      draw_features( img1, feat1, n1 );
      display_big_img( img1, argv[1] );
      cvWaitKey( 0 );
    }
  fprintf( stderr, "Finding features in %s...\n", argv[2] );
  n2 = sift_features( img2, &feat2 );
    if( display )
    {
      draw_features( img2, feat2, n2 );
      display_big_img( img2, argv[2] );
      cvWaitKey( 0 );
    }

	clock_t start, end;
	printf("特征匹配");
	start = clock();

	fprintf(stderr, "Building kd tree...\n");
	kd_root = kdtree_build(feat2, n2);



	//分配内存
	double src_x[10000] ;
	double src_y[10000];
	double drc_x[10000];
	double drc_y[10000];

	int w = 0;
	for (i = 0; i < n1; i++)	//逐点匹配
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);	//找2个最近点

		if (k == 2)
		{
			d0 = descr_dist_sq(feat, nbrs[0]);
			d1 = descr_dist_sq(feat, nbrs[1]);
			if (d0 < d1 * NN_SQ_DIST_RATIO_THR)	//最近点与次最近点距离之比要小才当做正确匹配，然后画一条线
			{
				pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
				pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
				src_x[w] = feat->x; 
				src_y[w] = feat->y; 
				drc_x[w] = nbrs[0]->x; 
				drc_y[w] = nbrs[0]->y; 
				w += 1;
				//带误匹配--紫红色
				//pt2.x += img1->width;
				//cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
				//m++;
				//feat1[i].fwd_match = nbrs[0];

				//剔除误匹配--绿色
				if (!(pt1.x == 413 && pt1.y == 177) && !(pt1.x == 499 && pt1.y == 33) && !(pt1.x == 400 && pt1.y == 239))
				{
				    //pt2.y-pt1.y>20
					pt2.x += img1->width;
					cvLine(stacked, pt1, pt2, CV_RGB(0, 255, 0), 1, 8, 0);
					m++;
					feat1[i].fwd_match = nbrs[0];
				}
			}
		}
		free(nbrs);
	}

	end = clock();
	printf("time(ms)=%f\n", (double)(end - start));




	printf("总时间");
	end22 = clock();
	printf("time(ms)=%f\n", (double)(end22 - start22));



  fprintf( stderr, "Found %d total matches\n", m );
  display_big_img( stacked, "Matches" );
  cvWaitKey( 0 );


  /* 
     UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS
     
     Note that this line above:
     
     feat1[i].fwd_match = nbrs[0];
     
     is important for the RANSAC function to work.
  */
  //计算变换参数
 // {
 //   CvMat* H;
 //   IplImage* xformed;
 //   H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
	//	      homog_xfer_err, 3.0, NULL, NULL );
 //   if( H )
 //     {
	//xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
	//cvWarpPerspective( img1, xformed, H, 
	//		   CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
	//		   cvScalarAll( 0 ) );
	//cvNamedWindow( "Xformed", 1 );
	//cvShowImage( "Xformed", xformed );
	//cvWaitKey( 0 );
	//cvReleaseImage( &xformed );
	//cvReleaseMat( &H );
 //     }
 // }
 // 

 // cvReleaseImage( &stacked );
 // cvReleaseImage( &img1 );
 // cvReleaseImage( &img2 );
 // kdtree_release( kd_root );
 // free( feat1 );
 // free( feat2 );
 // return 0;


}
