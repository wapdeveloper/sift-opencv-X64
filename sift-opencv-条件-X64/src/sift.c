/*
  Functions for detecting SIFT image features.
  
  For more information, refer to:
  
  Lowe, D.  Distinctive image features from scale-invariant keypoints.
  <EM>International Journal of Computer Vision, 60</EM>, 2 (2004),
  pp.91--110.
  
  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  Note: The SIFT algorithm is patented in the United States and cannot be
  used in commercial products without a license from the University of
  British Columbia.  For more information, refer to the file LICENSE.ubc
  that accompanied this distribution.

  @version 1.1.2-20100521
*/
#include <windows.h>
#include "sift.h"
#include "imgfeatures.h"
#include "utils.h"
#include "time.h"
#include <cxcore.h>
#include <cv.h>

/************************* Local Function Prototypes *************************/

static IplImage* create_init_img( IplImage*, int, double );
static IplImage* convert_to_gray32( IplImage* );
static IplImage*** build_gauss_pyr( IplImage*, int, int, double );
static IplImage* downsample( IplImage* );
static IplImage*** build_dog_pyr( IplImage***, int, int );
static CvSeq* scale_space_extrema( IplImage***, int, int, double, int,
				   CvMemStorage*);
static int is_extremum( IplImage***, int, int, int, int );
static struct feature* interp_extremum( IplImage***, int, int, int, int, int,
					double);
static void interp_step( IplImage***, int, int, int, int, double*, double*,
			 double* );
static CvMat* deriv_3D( IplImage***, int, int, int, int );
static CvMat* hessian_3D( IplImage***, int, int, int, int );
static double interp_contr( IplImage***, int, int, int, int, double, double,
			    double );
static struct feature* new_feature( void );
static int is_too_edge_like( IplImage*, int, int, int );
static void calc_feature_scales( CvSeq*, double, int );
static void adjust_for_img_dbl( CvSeq* );
static void calc_feature_oris( CvSeq*, IplImage*** );
static double* ori_hist( IplImage*, int, int, int, int, double );
static int calc_grad_mag_ori( IplImage*, int, int, double*, double* );
static void smooth_ori_hist( double*, int );
static double dominant_ori( double*, int );
static void add_good_ori_features( CvSeq*, double*, int, double,
				   struct feature* );
static struct feature* clone_feature( struct feature* );
static void compute_descriptors( CvSeq*, IplImage***, int, int );
static double*** descr_hist( IplImage*, int, int, double, double, int, int );
static void interp_hist_entry( double***, double, double, double, double, int,
			       int);
static void hist_to_descr( double***, int, int, struct feature* );
static void normalize_descr( struct feature* );
static int feature_cmp( void*, void*, void* );
static void release_descr_hist( double****, int );
static void release_pyr( IplImage****, int, int );


/*********************** Functions prototyped in sift.h **********************/


/**
   Finds SIFT features in an image using default parameter values.  All
   detected features are stored in the array pointed to by \a feat.

   @param img the image in which to detect features
   @param feat a pointer to an array in which to store detected features

   @return Returns the number of features stored in \a feat or -1 on failure
   @see _sift_features()
*/
int sift_features( IplImage* img, struct feature** feat )
{
  return _sift_features( img, feat, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,
			 SIFT_CURV_THR, SIFT_IMG_DBL, SIFT_DESCR_WIDTH,
			 SIFT_DESCR_HIST_BINS );
}



/**
   Finds SIFT features in an image using user-specified parameter values.  All
   detected features are stored in the array pointed to by \a feat.

   @param img the image in which to detect features
   @param fea a pointer to an array in which to store detected features
   @param intvls the number of intervals sampled per octave of scale space
   @param sigma the amount of Gaussian smoothing applied to each image level
     before building the scale space representation for an octave
   @param cont_thr a threshold on the value of the scale space function
     \f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
     feature location and scale, used to reject unstable features;  assumes
     pixel values in the range [0, 1]
   @param curv_thr threshold on a feature's ratio of principle curvatures
     used to reject features that are too edge-like
   @param img_dbl should be 1 if image doubling prior to scale space
     construction is desired or 0 if not
   @param descr_width the width, \f$n\f$, of the \f$n \times n\f$ array of
     orientation histograms used to compute a feature's descriptor
   @param descr_hist_bins the number of orientations in each of the
     histograms in the array used to compute a feature's descriptor

   @return Returns the number of keypoints stored in \a feat or -1 on failure
   @see sift_keypoints()
*/
int _sift_features( IplImage* img, struct feature** feat, int intvls,
		    double sigma, double contr_thr, int curv_thr,
		    int img_dbl, int descr_width, int descr_hist_bins )
{
  IplImage* init_img;
  IplImage*** gauss_pyr, *** dog_pyr;
  CvMemStorage* storage;
  CvSeq* features;
  int octvs, i, n = 0;
  
  /* check arguments */
  if( ! img )
    fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );
  if( ! feat )
    fatal_error( "NULL pointer error, %s, line %d",  __FILE__, __LINE__ );

  /* build scale space pyramid; smallest dimension of top level is ~4 pixels */
  init_img = create_init_img( img, img_dbl, sigma );	//创建初始图像
  octvs = log( MIN( init_img->width, init_img->height ) ) / log(2) - 2;	//看一共能分几组(Octave)

  printf("高斯金字塔");
  clock_t start, end;
  start = clock();
  gauss_pyr = build_gauss_pyr( init_img, octvs, intvls, sigma );	//创建高斯金字塔
  end = clock();
  printf("time(ms)=%f\n", (double)(end - start) );







  dog_pyr = build_dog_pyr( gauss_pyr, octvs, intvls );	//创建高斯差分金字塔
  
  storage = cvCreateMemStorage( 0 );


  printf("极值点检测");
  start = clock();
  features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr,
	  curv_thr, storage);	//在高斯差分金字塔中寻找尺度空间极值点
  end = clock();
  printf("time(ms)=%f\n", (double)(end - start));






  calc_feature_scales( features, sigma, intvls );	//计算特征点处的尺度scale，相当于高斯函数的那个sigma
  if( img_dbl )
    adjust_for_img_dbl( features );	//对于初始加倍的图像，一些参数除以2，以获得相对于最原始图像的参数

  printf("生成主方向");
  start = clock();
  calc_feature_oris(features, gauss_pyr);	//给特征点分配方向参数
  end = clock();
  printf("time(ms)=%f\n", (double)(end - start));


  printf("生成描述符");
  start = clock();
  compute_descriptors(features, gauss_pyr, descr_width, descr_hist_bins);	//计算局部图像描述子
  end = clock();
  printf("time(ms)=%f\n", (double)(end - start));






  /* sort features by decreasing scale and move from CvSeq to array */
  cvSeqSort( features, (CvCmpFunc)feature_cmp, NULL );
  n = features->total;
  *feat = calloc( n, sizeof(struct feature) );
  *feat = cvCvtSeqToArray( features, *feat, CV_WHOLE_SEQ );
  for( i = 0; i < n; i++ )
    {
      free( (*feat)[i].feature_data );
      (*feat)[i].feature_data = NULL;
    }
  
  cvReleaseMemStorage( &storage );
  cvReleaseImage( &init_img );
  release_pyr( &gauss_pyr, octvs, intvls + 3 );
  release_pyr( &dog_pyr, octvs, intvls + 2 );
  return n;
}


/************************ Functions prototyped here **************************/

/*
  Converts an image to 8-bit grayscale and Gaussian-smooths it.  The image is
  optionally doubled in size prior to smoothing.

  @param img input image
  @param img_dbl if true, image is doubled in size prior to smoothing
  @param sigma total std of Gaussian smoothing
*/
static IplImage* create_init_img( IplImage* img, int img_dbl, double sigma )
{
  IplImage* gray, * dbl;
  double sig_diff;

  gray = convert_to_gray32( img );	//转成灰度图
  if( img_dbl )	//初始图像加倍的情况
    {
      sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
	  /*由于大的高斯模糊的半径是所用多个高斯模糊半径平方和的平方根，而放大图像初始就有
	  2*SIFT_INIT_SIGMA的模糊度，为达到最终sigma的模糊度，只需再叠加sig_diff的高斯模糊*/
      dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ),
			   IPL_DEPTH_32F, 1 );	//db1为原始图像的两倍尺寸
      cvResize( gray, dbl, CV_INTER_CUBIC );	//采用立方插值放大原始图像
      cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );	//高斯模糊
      cvReleaseImage( &gray );
      return dbl;
    }
  else
    {
      sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );
      cvSmooth( gray, gray, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff );
      return gray;
    }
}



/*
  Converts an image to 32-bit grayscale

  @param img a 3-channel 8-bit color (BGR) or 8-bit gray image

  @return Returns a 32-bit grayscale image
*/
static IplImage* convert_to_gray32( IplImage* img )
{
  IplImage* gray8, * gray32;
  
  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  if( img->nChannels == 1 )
    gray8 = cvClone( img );
  else
    {
      gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
      cvCvtColor( img, gray8, CV_BGR2GRAY );
    }
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}



/*
  Builds Gaussian scale space pyramid from an image

  @param base base image of the pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave
  @param sigma amount of Gaussian smoothing per octave

  @return Returns a Gaussian scale space pyramid as an octvs x (intvls + 3)
    array
*/
static IplImage*** build_gauss_pyr( IplImage* base, int octvs,
			     int intvls, double sigma )
{
  IplImage*** gauss_pyr;
  double *sig = (double *)calloc(intvls + 3,sizeof(double));	//每组需s+3张图像，默认6张,sig存图像对应的sigma值，各组通用
  double sig_total, sig_prev, k;
  int i, o;

  gauss_pyr = calloc( octvs, sizeof( IplImage** ) );	//octvs组图像
  for( i = 0; i < octvs; i++ )
    gauss_pyr[i] = calloc( intvls + 3, sizeof( IplImage *) );	//每组需s+3张图像，默认6张

  /*
    precompute Gaussian sigmas using the following formula:

    \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
  */
  sig[0] = sigma;	//初始sigma，默认1.6
  k = pow( 2.0, 1.0 / intvls );	//k=2^(1/s)
  for( i = 1; i < intvls + 3; i++ )
    {
      sig_prev = pow( k, i - 1 ) * sigma;	//前一幅图像的sigma
      sig_total = sig_prev * k;				//当前图像的sigma
      sig[i] = sqrt( sig_total * sig_total - sig_prev * sig_prev );	//需叠加的高斯模糊参数值，应该叫sig_diff更贴切
    }

  for( o = 0; o < octvs; o++ )
    for( i = 0; i < intvls + 3; i++ )
      {
	if( o == 0  &&  i == 0 )	//高斯金字塔最低层
	  gauss_pyr[o][i] = cvCloneImage(base);

	/* base of new octvave is halved image from end of previous octave */
	else if( i == 0 )	//每一组最底层由上一组对应图像下采样获得。其实除了起始组外，每一组的前3幅图像都可以由上组下采样获得
	  gauss_pyr[o][i] = downsample( gauss_pyr[o-1][intvls] );
	  
	/* blur the current octave's last image to create the next one */
	else  //在同组上层图像基础上叠加sig_diff得到本层图像
	  {
	    gauss_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i-1]),
					     IPL_DEPTH_32F, 1 );
	    cvSmooth( gauss_pyr[o][i-1], gauss_pyr[o][i],
		      CV_GAUSSIAN, 0, 0, sig[i], sig[i] );
	  }
      }
  free(sig);
  return gauss_pyr;
}



/*
  Downsamples an image to a quarter of its size (half in each dimension)
  using nearest-neighbor interpolation

  @param img an image

  @return Returns an image whose dimensions are half those of img
*/
static IplImage* downsample( IplImage* img )
{
  IplImage* smaller = cvCreateImage( cvSize(img->width / 2, img->height / 2),
				     img->depth, img->nChannels );
  cvResize( img, smaller, CV_INTER_NN );

  return smaller;
}



/*
  Builds a difference of Gaussians scale space pyramid by subtracting adjacent
  intervals of a Gaussian pyramid

  @param gauss_pyr Gaussian scale-space pyramid
  @param octvs number of octaves of scale space
  @param intvls number of intervals per octave

  @return Returns a difference of Gaussians scale space pyramid as an
    octvs x (intvls + 2) array
*/
static IplImage*** build_dog_pyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
  IplImage*** dog_pyr;
  int i, o;

  dog_pyr = calloc( octvs, sizeof( IplImage** ) );
  for( i = 0; i < octvs; i++ )
    dog_pyr[i] = calloc( intvls + 2, sizeof(IplImage*) );	//高斯差分金字塔每组只需s+2幅图像，默认为5张

  for( o = 0; o < octvs; o++ )
    for( i = 0; i < intvls + 2; i++ )
      {
	dog_pyr[o][i] = cvCreateImage( cvGetSize(gauss_pyr[o][i]),
				       IPL_DEPTH_32F, 1 );
	cvSub( gauss_pyr[o][i+1], gauss_pyr[o][i], dog_pyr[o][i], NULL );	//为高斯金字塔两层相减获得
      }

  return dog_pyr;
}



/*
  Detects features at extrema in DoG scale space.  Bad features are discarded
  based on contrast and ratio of principal curvatures.

  @param dog_pyr DoG scale space pyramid
  @param octvs octaves of scale space represented by dog_pyr
  @param intvls intervals per octave
  @param contr_thr low threshold on feature contrast
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  @return Returns an array of detected features whose scales, orientations,
    and descriptors are yet to be determined.
*/
static CvSeq* scale_space_extrema( IplImage*** dog_pyr, int octvs, int intvls,
				   double contr_thr, int curv_thr,
				   CvMemStorage* storage )
{
  CvSeq* features;
  double prelim_contr_thr = 0.5 * contr_thr / intvls;	//分层多会导致直接差值得到的值变小，因此这里对阈值进行调整
  struct feature* feat;
  struct detection_data* ddata;
  int o, i, r, c;

  features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );
  for( o = 0; o < octvs; o++ )
    for( i = 1; i <= intvls; i++ )
      for(r = SIFT_IMG_BORDER; r < dog_pyr[o][0]->height-SIFT_IMG_BORDER; r++)	//忽略边框附近像素
	for(c = SIFT_IMG_BORDER; c < dog_pyr[o][0]->width-SIFT_IMG_BORDER; c++)
	  /* perform preliminary check on contrast */
	  if( ABS( pixval32f( dog_pyr[o][i], r, c ) ) > prelim_contr_thr )	//要求高对比度
	    if( is_extremum( dog_pyr, o, i, r, c ) )
	      {
		feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);	//亚像素精度，并排除低对比度点
		if( feat )
		  {
		    ddata = feat_detection_data( feat );
		    if( ! is_too_edge_like( dog_pyr[ddata->octv][ddata->intvl],
					    ddata->r, ddata->c, curv_thr ) )	//去除r太大的点
		      {
			cvSeqPush( features, feat );
		      }
		    else
		      free( ddata );
		    free( feat );
		  }
	      }
  
  return features;
}



/*
  Determines whether a pixel is a scale-space extremum by comparing it to it's
  3x3x3 pixel neighborhood.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's scale space octave
  @param intvl pixel's within-octave interval
  @param r pixel's image row
  @param c pixel's image col

  @return Returns 1 if the specified pixel is an extremum (max or min) among
    it's 3x3x3 pixel neighborhood.
*/
static int is_extremum( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
  double val = pixval32f( dog_pyr[octv][intvl], r, c );
  int i, j, k;

  /* check for maximum */
  if( val > 0 )
    {
      for( i = -1; i <= 1; i++ )
	for( j = -1; j <= 1; j++ )
	  for( k = -1; k <= 1; k++ )
	    if( val < pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
	      return 0;
    }

  /* check for minimum */
  else
    {
      for( i = -1; i <= 1; i++ )
	for( j = -1; j <= 1; j++ )
	  for( k = -1; k <= 1; k++ )
	    if( val > pixval32f( dog_pyr[octv][intvl+i], r + j, c + k ) )
	      return 0;
    }

  return 1;
}



/*
  Interpolates a scale-space extremum's location and scale to subpixel
  accuracy to form an image feature.  Rejects features with low contrast.
  Based on Section 4 of Lowe's paper.  

  @param dog_pyr DoG scale space pyramid
  @param octv feature's octave of scale space
  @param intvl feature's within-octave interval
  @param r feature's image row
  @param c feature's image column
  @param intvls total intervals per octave
  @param contr_thr threshold on feature contrast

  @return Returns the feature resulting from interpolation of the given
    parameters or NULL if the given location could not be interpolated or
    if contrast at the interpolated loation was too low.  If a feature is
    returned, its scale, orientation, and descriptor are yet to be determined.
*/
static struct feature* interp_extremum( IplImage*** dog_pyr, int octv,
					int intvl, int r, int c, int intvls,
					double contr_thr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double xi, xr, xc, contr;
  int i = 0;
  
  while( i < SIFT_MAX_INTERP_STEPS )
    {
      interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );	//计算出off_X(x,y,sigma)
      if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
	break;	//任一维都小于0.5才能结束，否则需换点重算，重算次数有限制
      
      c += cvRound( xc );
      r += cvRound( xr );
      intvl += cvRound( xi );
      
      if( intvl < 1  ||
	  intvl > intvls  ||
	  c < SIFT_IMG_BORDER  ||
	  r < SIFT_IMG_BORDER  ||
	  c >= dog_pyr[octv][0]->width - SIFT_IMG_BORDER  ||
	  r >= dog_pyr[octv][0]->height - SIFT_IMG_BORDER )
	{
	  return NULL;
	}
      
      i++;
    }
  
  /* ensure convergence of interpolation */
  if( i >= SIFT_MAX_INTERP_STEPS )
    return NULL;
  
  contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );	//计算新极值点的D(x)
  if( ABS( contr ) < contr_thr / intvls )
    return NULL;
  //满足条件，存储这个特征点
  feat = new_feature();
  ddata = feat_detection_data( feat );
  feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );	//乘以2^octv以得到对应于底层图像的位置
  feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
  ddata->r = r;	//行
  ddata->c = c;	//列
  ddata->octv = octv;
  ddata->intvl = intvl;	//层
  ddata->subintvl = xi;	//off_sigma

  return feat;
}



/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl interval being interpolated
  @param r row being interpolated
  @param c column being interpolated
  @param xi output as interpolated subpixel increment to interval
  @param xr output as interpolated subpixel increment to row
  @param xc output as interpolated subpixel increment to col
*/

static void interp_step( IplImage*** dog_pyr, int octv, int intvl, int r, int c,
			 double* xi, double* xr, double* xc )
{
  CvMat* dD, * H, * H_inv, X;
  double x[3] = { 0 };
  
  dD = deriv_3D( dog_pyr, octv, intvl, r, c ); //计算了dD/dx,dD/dy,dD/d(sigma),返回为列向量dD/dX
  H = hessian_3D( dog_pyr, octv, intvl, r, c );	//用离散值近似计算出三维hessian矩阵,即公式中d2D/dX2
  H_inv = cvCreateMat( 3, 3, CV_64FC1 );
  cvInvert( H, H_inv, CV_SVD );		//矩阵求逆
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 ); //计算结果为-(d2D/dX2)^(-1)*(dD/dX),即off_X
  
  cvReleaseMat( &dD );
  cvReleaseMat( &H );
  cvReleaseMat( &H_inv );

  *xi = x[2];
  *xr = x[1];
  *xc = x[0];
}



/*
  Computes the partial derivatives in x, y, and scale of a pixel in the DoG
  scale space pyramid.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the vector of partial derivatives for pixel I
    { dI/dx, dI/dy, dI/ds }^T as a CvMat*
*/
static CvMat* deriv_3D( IplImage*** dog_pyr, int octv, int intvl, int r, int c )
{
  CvMat* dI;
  double dx, dy, ds;

  dx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) -
	 pixval32f( dog_pyr[octv][intvl], r, c-1 ) ) / 2.0;
  dy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) -
	 pixval32f( dog_pyr[octv][intvl], r-1, c ) ) / 2.0;
  ds = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) -
	 pixval32f( dog_pyr[octv][intvl-1], r, c ) ) / 2.0;
  
  dI = cvCreateMat( 3, 1, CV_64FC1 );
  cvmSet( dI, 0, 0, dx );
  cvmSet( dI, 1, 0, dy );
  cvmSet( dI, 2, 0, ds );

  return dI;
}



/*
  Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.

  @param dog_pyr DoG scale space pyramid
  @param octv pixel's octave in dog_pyr
  @param intvl pixel's interval in octv
  @param r pixel's image row
  @param c pixel's image col

  @return Returns the Hessian matrix (below) for pixel I as a CvMat*

  / Ixx  Ixy  Ixs \ <BR>
  | Ixy  Iyy  Iys | <BR>
  \ Ixs  Iys  Iss /
*/
static CvMat* hessian_3D( IplImage*** dog_pyr, int octv, int intvl, int r,
			  int c )
{
  CvMat* H;
  double v, dxx, dyy, dss, dxy, dxs, dys;
  
  v = pixval32f( dog_pyr[octv][intvl], r, c );
  dxx = ( pixval32f( dog_pyr[octv][intvl], r, c+1 ) + 
	  pixval32f( dog_pyr[octv][intvl], r, c-1 ) - 2 * v );
  dyy = ( pixval32f( dog_pyr[octv][intvl], r+1, c ) +
	  pixval32f( dog_pyr[octv][intvl], r-1, c ) - 2 * v );
  dss = ( pixval32f( dog_pyr[octv][intvl+1], r, c ) +
	  pixval32f( dog_pyr[octv][intvl-1], r, c ) - 2 * v );
  dxy = ( pixval32f( dog_pyr[octv][intvl], r+1, c+1 ) -
	  pixval32f( dog_pyr[octv][intvl], r+1, c-1 ) -
	  pixval32f( dog_pyr[octv][intvl], r-1, c+1 ) +
	  pixval32f( dog_pyr[octv][intvl], r-1, c-1 ) ) / 4.0;
  dxs = ( pixval32f( dog_pyr[octv][intvl+1], r, c+1 ) -
	  pixval32f( dog_pyr[octv][intvl+1], r, c-1 ) -
	  pixval32f( dog_pyr[octv][intvl-1], r, c+1 ) +
	  pixval32f( dog_pyr[octv][intvl-1], r, c-1 ) ) / 4.0;
  dys = ( pixval32f( dog_pyr[octv][intvl+1], r+1, c ) -
	  pixval32f( dog_pyr[octv][intvl+1], r-1, c ) -
	  pixval32f( dog_pyr[octv][intvl-1], r+1, c ) +
	  pixval32f( dog_pyr[octv][intvl-1], r-1, c ) ) / 4.0;
  
  H = cvCreateMat( 3, 3, CV_64FC1 );
  cvmSet( H, 0, 0, dxx );
  cvmSet( H, 0, 1, dxy );
  cvmSet( H, 0, 2, dxs );
  cvmSet( H, 1, 0, dxy );
  cvmSet( H, 1, 1, dyy );
  cvmSet( H, 1, 2, dys );
  cvmSet( H, 2, 0, dxs );
  cvmSet( H, 2, 1, dys );
  cvmSet( H, 2, 2, dss );

  return H;
}



/*
  Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's
  paper.

  @param dog_pyr difference of Gaussians scale space pyramid
  @param octv octave of scale space
  @param intvl within-octave interval
  @param r pixel row
  @param c pixel column
  @param xi interpolated subpixel increment to interval
  @param xr interpolated subpixel increment to row
  @param xc interpolated subpixel increment to col

  @param Returns interpolated contrast.
*/
static double interp_contr( IplImage*** dog_pyr, int octv, int intvl, int r,
			    int c, double xi, double xr, double xc )
{
  CvMat* dD, X, T;
  double t[1], x[3] = { xc, xr, xi };

  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
  dD = deriv_3D( dog_pyr, octv, intvl, r, c );	//dD/dX
  cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );	//(dD^T/dX)*off_X
  cvReleaseMat( &dD );

  return pixval32f( dog_pyr[octv][intvl], r, c ) + t[0] * 0.5; //D+(dD^T/dX)*off_X/2
}



/*
  Allocates and initializes a new feature

  @return Returns a pointer to the new feature
*/
static struct feature* new_feature( void )
{
  struct feature* feat;
  struct detection_data* ddata;

  feat = malloc( sizeof( struct feature ) );
  memset( feat, 0, sizeof( struct feature ) );
  ddata = malloc( sizeof( struct detection_data ) );
  memset( ddata, 0, sizeof( struct detection_data ) );
  feat->feature_data = ddata;
  feat->type = FEATURE_LOWE;

  return feat;
}



/*
  Determines whether a feature is too edge like to be stable by computing the
  ratio of principal curvatures at that feature.  Based on Section 4.1 of
  Lowe's paper.

  @param dog_img image from the DoG pyramid in which feature was detected
  @param r feature row
  @param c feature col
  @param curv_thr high threshold on ratio of principal curvatures

  @return Returns 0 if the feature at (r,c) in dog_img is sufficiently
    corner-like or 1 otherwise.
*/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
  double d, dxx, dyy, dxy, tr, det;

  /* principal curvatures are computed using the trace and det of Hessian */
  d = pixval32f(dog_img, r, c);
  dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
  dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
  dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) -
	  pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
  tr = dxx + dyy;
  det = dxx * dyy - dxy * dxy;

  /* negative determinant -> curvatures have different signs; reject feature */
  if( det <= 0 )
    return 1;

  if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
    return 0;
  return 1;
}



/*
  Calculates characteristic scale for each feature in an array.

  @param features array of features
  @param sigma amount of Gaussian smoothing per octave of scale space
  @param intvls intervals per octave of scale space
*/
static void calc_feature_scales( CvSeq* features, double sigma, int intvls )
{
  struct feature* feat;
  struct detection_data* ddata;
  double intvl;
  int i, n;

  n = features->total;
  for( i = 0; i < n; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      ddata = feat_detection_data( feat );
      intvl = ddata->intvl + ddata->subintvl;
      feat->scl = sigma * pow( 2.0, ddata->octv + intvl / intvls );
      ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}



/*
  Halves feature coordinates and scale in case the input image was doubled
  prior to scale space construction.

  @param features array of features
*/
static void adjust_for_img_dbl( CvSeq* features )
{
  struct feature* feat;
  int i, n;

  n = features->total;
  for( i = 0; i < n; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      feat->x /= 2.0;
      feat->y /= 2.0;
      feat->scl /= 2.0;
      feat->img_pt.x /= 2.0;
      feat->img_pt.y /= 2.0;
    }
}



/*
  Computes a canonical orientation for each image feature in an array.  Based
  on Section 5 of Lowe's paper.  This function adds features to the array when
  there is more than one dominant orientation at a given feature location.

  @param features an array of image features
  @param gauss_pyr Gaussian scale space pyramid
*/
static void calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double* hist;
  double omax;
  int i, j, n = features->total;

  for( i = 0; i < n; i++ )
    {
      feat = malloc( sizeof( struct feature ) );
      cvSeqPopFront( features, feat );
      ddata = feat_detection_data( feat );
      hist = ori_hist( gauss_pyr[ddata->octv][ddata->intvl],	//生成方向分配所需直方图
		       ddata->r, ddata->c, SIFT_ORI_HIST_BINS,
		       cvRound( SIFT_ORI_RADIUS * ddata->scl_octv ),	//计算区域半径3*1.5*sigma
		       SIFT_ORI_SIG_FCTR * ddata->scl_octv );
      for( j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
	smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );	//直方图平滑
      omax = dominant_ori( hist, SIFT_ORI_HIST_BINS );	//直方图峰值
      add_good_ori_features( features, hist, SIFT_ORI_HIST_BINS,	//增加其他方向并进行二次插值
			     omax * SIFT_ORI_PEAK_RATIO, feat );
      free( ddata );
      free( feat );
      free( hist );
    }
}



/*
  Computes a gradient orientation histogram at a specified pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param n number of histogram bins
  @param rad radius of region over which histogram is computed
  @param sigma std for Gaussian weighting of histogram entries

  @return Returns an n-element array containing an orientation histogram
    representing orientations between 0 and 2 PI.
*/
static double* ori_hist( IplImage* img, int r, int c, int n, int rad,
			 double sigma )
{
  double* hist;
  double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
  int bin, i, j;

  hist = calloc( n, sizeof( double ) );
  exp_denom = 2.0 * sigma * sigma;
  for( i = -rad; i <= rad; i++ )
    for( j = -rad; j <= rad; j++ )
      if( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )	//计算这点的梯度
	{
	  w = exp( -( i*i + j*j ) / exp_denom );	//w为高斯权重系数
	  bin = cvRound( n * ( ori + CV_PI ) / PI2 );	//角度值映射至0~35，四舍五入
	  bin = ( bin < n )? bin : 0;
	  hist[bin] += w * mag;	//乘以幅值权重
	}

  return hist;
}



/*
  Calculates the gradient magnitude and orientation at a given pixel.

  @param img image
  @param r pixel row
  @param c pixel col
  @param mag output as gradient magnitude at pixel (r,c)
  @param ori output as gradient orientation at pixel (r,c)

  @return Returns 1 if the specified pixel is a valid one and sets mag and
    ori accordingly; otherwise returns 0
*/
static int calc_grad_mag_ori( IplImage* img, int r, int c, double* mag,
			      double* ori )
{
  double dx, dy;

  if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 )
    {
      dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
      dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
      *mag = sqrt( dx*dx + dy*dy );
      *ori = atan2( dy, dx );
      return 1;
    }

  else
    return 0;
}



/*
  Gaussian smooths an orientation histogram.

  @param hist an orientation histogram
  @param n number of bins
*/
static void smooth_ori_hist( double* hist, int n )
{
  double prev, tmp, h0 = hist[0];
  int i;

  prev = hist[n-1];
  for( i = 0; i < n; i++ )
    {
      tmp = hist[i];
      hist[i] = 0.25 * prev + 0.5 * hist[i] + 
	0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
      prev = tmp;
    }
}



/*
  Finds the magnitude of the dominant orientation in a histogram

  @param hist an orientation histogram
  @param n number of bins

  @return Returns the value of the largest bin in hist
*/
static double dominant_ori( double* hist, int n )
{
  double omax;
  int maxbin, i;

  omax = hist[0];
  maxbin = 0;
  for( i = 1; i < n; i++ )
    if( hist[i] > omax )
      {
	omax = hist[i];
	maxbin = i;
      }
  return omax;
}



/*
  Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )



/*
  Adds features to an array for every orientation in a histogram greater than
  a specified threshold.

  @param features new features are added to the end of this array
  @param hist orientation histogram
  @param n number of bins in hist
  @param mag_thr new features are added for entries in hist greater than this
  @param feat new features are clones of this with different orientations
*/
static void add_good_ori_features( CvSeq* features, double* hist, int n,
				   double mag_thr, struct feature* feat )
{
  struct feature* new_feat;
  double bin, PI2 = CV_PI * 2.0;
  int l, r, i;

  for( i = 0; i < n; i++ )
    {
      l = ( i == 0 )? n - 1 : i-1;
      r = ( i + 1 ) % n;
      
      if( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
	{
	  bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );	//二次插值
	  bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
	  new_feat = clone_feature( feat );
	  new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI;	//恢复成对应弧度制角度
	  cvSeqPush( features, new_feat );
	  free( new_feat );
	}
    }
}



/*
  Makes a deep copy of a feature

  @param feat feature to be cloned

  @return Returns a deep copy of feat
*/
static struct feature* clone_feature( struct feature* feat )
{
  struct feature* new_feat;
  struct detection_data* ddata;

  new_feat = new_feature();
  ddata = feat_detection_data( new_feat );
  memcpy( new_feat, feat, sizeof( struct feature ) );
  memcpy( ddata, feat_detection_data(feat), sizeof( struct detection_data ) );
  new_feat->feature_data = ddata;

  return new_feat;
}



/*
  Computes feature descriptors for features in an array.  Based on Section 6
  of Lowe's paper.

  @param features array of features
  @param gauss_pyr Gaussian scale space pyramid
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void compute_descriptors( CvSeq* features, IplImage*** gauss_pyr, int d,
				 int n )
{
  struct feature* feat;
  struct detection_data* ddata;
  double*** hist;
  int i, k = features->total;

  for( i = 0; i < k; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      ddata = feat_detection_data( feat );
      hist = descr_hist( gauss_pyr[ddata->octv][ddata->intvl], ddata->r,
			 ddata->c, feat->ori, ddata->scl_octv, d, n );	//计算描述子
      hist_to_descr( hist, d, n, feat );
      release_descr_hist( &hist, d );
    }
}



/*
  Computes the 2D array of orientation histograms that form the feature
  descriptor.  Based on Section 6.1 of Lowe's paper.

  @param img image used in descriptor computation
  @param r row coord of center of orientation histogram array
  @param c column coord of center of orientation histogram array
  @param ori canonical orientation of feature whose descr is being computed
  @param scl scale relative to img of feature whose descr is being computed
  @param d width of 2d array of orientation histograms
  @param n bins per orientation histogram

  @return Returns a d x d array of n-bin orientation histograms.
*/
static double*** descr_hist( IplImage* img, int r, int c, double ori,
			     double scl, int d, int n )
{
  double*** hist;
  double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag,
    grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
  int radius, i, j;

  hist = calloc( d, sizeof( double** ) );	//共有d*d个直方图，每个直方图包含n个数据
  for( i = 0; i < d; i++ )
    {
      hist[i] = calloc( d, sizeof( double* ) );
      for( j = 0; j < d; j++ )
	hist[i][j] = calloc( n, sizeof( double ) );
    }
  
  cos_t = cos( ori );
  sin_t = sin( ori );
  bins_per_rad = n / PI2;	//直方图每份覆盖角度范围
  exp_denom = d * d * 0.5;	//高斯系数取为描述子宽度一半。本来应该为(0.5*d*hist_width)^2更好理解，这是为了后面w计算方便
  hist_width = SIFT_DESCR_SCL_FCTR * scl;	//3sigma，scl为关键点的尺度.
  radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;	//描述子覆盖区域半径，最后加0.5以实现四舍五入
  for( i = -radius; i <= radius; i++ )	//包含在大实线框中的点都需要判断
    for( j = -radius; j <= radius; j++ )
      {
	/*
	  Calculate sample's histogram array coords rotated relative to ori.
	  Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
	  r_rot = 1.5) have full weight placed in row 1 after interpolation.
	*/
	c_rot = ( j * cos_t - i * sin_t ) / hist_width;//计算在旋转坐标系下的坐标表示（点不动，坐标轴转），以小区域宽度为步长单位
	r_rot = ( j * sin_t + i * cos_t ) / hist_width;
	rbin = r_rot + d / 2 - 0.5;	//rbin是在新坐标系下的坐标，这里单位长度为hist_width，这里又进行平移，使原点变到（1.5，1.5)
	cbin = c_rot + d / 2 - 0.5; //加d/2后将小实线框的左上角平移至原点，再减0.5又回去一点
	
	//这里是看这个点是否在虚线框内。虚线框宽度在这个坐标系下为d+1，如果rbin不减0.5就更容易理解了，这样rbin>-0.5 && rbin<d+0.5，rbin的取值范围恰是d+1
	if( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )	
	  if( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori )) //计算此点的梯度方向，并化成相对于关键点方向的角度值。grad_mag梯度幅值
	    {
	      grad_ori -= ori;
	      while( grad_ori < 0.0 )
		grad_ori += PI2;
	      while( grad_ori >= PI2 )
		grad_ori -= PI2;
	      
	      obin = grad_ori * bins_per_rad;	//角度值映射至0~7之间
	      w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom ); //计算以关键点为中心的高斯权重系数,旋转对称，旋转不影响其值。
		  //由此公式得来w=exp(-(i^2+j^2)/(2*(0.5*d*hist_width)^2))=exp(-(c_rot^2+r_cot^2)/(0.5*d*d))
	      interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n ); //通过双线性插值计算直方图
	    }
      }

  return hist;
}



/*
  Interpolates an entry into the array of orientation histograms that form
  the feature descriptor.

  @param hist 2D array of orientation histograms
  @param rbin sub-bin row coordinate of entry
  @param cbin sub-bin column coordinate of entry
  @param obin sub-bin orientation coordinate of entry
  @param mag size of entry
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void interp_hist_entry( double*** hist, double rbin, double cbin,
			       double obin, double mag, int d, int n )
{
  double d_r, d_c, d_o, v_r, v_c, v_o;
  double** row, * h;
  int r0, c0, o0, rb, cb, ob, r, c, o;

  r0 = cvFloor( rbin );	//返回不大于参数的最大整数值
  c0 = cvFloor( cbin );
  o0 = cvFloor( obin );
  d_r = rbin - r0;	//得到小数部分
  d_c = cbin - c0;
  d_o = obin - o0;

  /*
    The entry is distributed into up to 8 bins.  Each entry into a bin
    is multiplied by a weight of 1 - d for each dimension, where d is the
    distance from the center value of the bin measured in bin units.
  */
  //这里也可以看出前面rbin、cbin为何减0.5。这样原点上这点的d_r、d_c均为0.5，即原点上这点的方向将被平均分配叠加在它
  //周围4个直方图上面
  for( r = 0; r <= 1; r++ )
    {
      rb = r0 + r;
      if( rb >= 0  &&  rb < d )
	{
	  v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );	//公式中的1-d_r
	  row = hist[rb];	//第rb行上的hist
	  for( c = 0; c <= 1; c++ )
	    {
	      cb = c0 + c;
	      if( cb >= 0  &&  cb < d )
		{
		  v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
		  h = row[cb]; //h表示第rb行cb列上的hist
		  for( o = 0; o <= 1; o++ )
		    {
		      ob = ( o0 + o ) % n;
		      v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );	//角度不是像ori_hist中那样直接四舍五入，而是分配到距它最近的两个方向上
		      h[ob] += v_o;
		    }
		}
	    }
	}
    }
}



/*
  Converts the 2D array of orientation histograms into a feature's descriptor
  vector.
  
  @param hist 2D array of orientation histograms
  @param d width of hist
  @param n bins per histogram
  @param feat feature into which to store descriptor
*/
static void hist_to_descr( double*** hist, int d, int n, struct feature* feat )
{
  int int_val, i, r, c, o, k = 0;

  for( r = 0; r < d; r++ )
    for( c = 0; c < d; c++ )
      for( o = 0; o < n; o++ )
	feat->descr[k++] = hist[r][c][o];

  feat->d = k;
  normalize_descr( feat );
  for( i = 0; i < k; i++ )
    if( feat->descr[i] > SIFT_DESCR_MAG_THR )
      feat->descr[i] = SIFT_DESCR_MAG_THR;
  normalize_descr( feat );

  /* convert floating-point descriptor to integer valued descriptor */
  for( i = 0; i < k; i++ )
    {
      int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
      feat->descr[i] = MIN( 255, int_val );
    }
}



/*
  Normalizes a feature's descriptor vector to unitl length

  @param feat feature
*/
static void normalize_descr( struct feature* feat )
{
  double cur, len_inv, len_sq = 0.0;
  int i, d = feat->d;

  for( i = 0; i < d; i++ )
    {
      cur = feat->descr[i];
      len_sq += cur*cur;
    }
  len_inv = 1.0 / sqrt( len_sq );
  for( i = 0; i < d; i++ )
    feat->descr[i] *= len_inv;
}



/*
  Compares features for a decreasing-scale ordering.  Intended for use with
  CvSeqSort

  @param feat1 first feature
  @param feat2 second feature
  @param param unused

  @return Returns 1 if feat1's scale is greater than feat2's, -1 if vice versa,
    and 0 if their scales are equal
*/
static int feature_cmp( void* feat1, void* feat2, void* param )
{
  struct feature* f1 = (struct feature*) feat1;
  struct feature* f2 = (struct feature*) feat2;

  if( f1->scl < f2->scl )
    return 1;
  if( f1->scl > f2->scl )
    return -1;
  return 0;
}



/*
  De-allocates memory held by a descriptor histogram

  @param hist pointer to a 2D array of orientation histograms
  @param d width of hist
*/
static void release_descr_hist( double**** hist, int d )
{
  int i, j;

  for( i = 0; i < d; i++)
    {
      for( j = 0; j < d; j++ )
	free( (*hist)[i][j] );
      free( (*hist)[i] );
    }
  free( *hist );
  *hist = NULL;
}


/*
  De-allocates memory held by a scale space pyramid

  @param pyr scale space pyramid
  @param octvs number of octaves of scale space
  @param n number of images per octave
*/
static void release_pyr( IplImage**** pyr, int octvs, int n )
{
  int i, j;
  for( i = 0; i < octvs; i++ )
    {
      for( j = 0; j < n; j++ )
	cvReleaseImage( &(*pyr)[i][j] );
      free( (*pyr)[i] );
    }
  free( *pyr );
  *pyr = NULL;
}
