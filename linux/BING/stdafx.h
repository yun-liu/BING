
#pragma once

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>

#ifdef _WIN32
#include <crtdbg.h>
#include <SDKDDKVer.h>
#endif

#pragma once
#pragma warning(disable: 4267)
#pragma warning(disable: 4805)
#pragma warning(disable: 4819)
#pragma warning(disable: 4995)
#pragma warning(disable: 4996)

#include <limits>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <time.h>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef __APPLE__
#include <omp.h>
#endif

#ifndef _WIN32
#include <sys/stat.h>
#include <dirent.h>
#endif

using namespace std;

// TODO: reference additional headers your program requires here
#include <opencv2/opencv.hpp> 
#include <opencv/cv.h>

using namespace cv;


#ifdef _WIN32
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#pragma comment( lib, cvLIB("ts"))
#pragma comment( lib, cvLIB("world"))
#endif


typedef const Mat CMat;
typedef const string CStr;
typedef vector<Mat> vecM;
typedef vector<string> vecS;
typedef vector<int> vecI;
typedef vector<bool> vecB;
typedef vector<float> vecF;
typedef vector<double> vecD;

#ifndef _WIN32
typedef uint64_t UINT64;
typedef uint32_t UINT;
typedef int64_t INT64;
typedef uint8_t byte;
typedef bool BOOL;

#ifndef FALSE
#define FALSE false
#endif
#define __popcnt64 __builtin_popcountll
#endif

enum{CV_FLIP_BOTH = -1, CV_FLIP_VERTICAL = 0, CV_FLIP_HORIZONTAL = 1};
#define _S(str) ((str).c_str())
#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)
#define CV_Assert_(expr, args) \
{\
	if (!(expr)) {\
	  string msg(args); \
	  printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
	  cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__) ); }\
}

// Return -1 if not in the list
template<typename T>
static inline int findFromList(const T &word, const vector<T> &strList) {size_t idx = find(strList.begin(), strList.end(), word) - strList.begin(); return idx < strList.size() ? (int)idx : -1;}
template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...
template<class T, int D> inline T vecSqrDist(const Vec<T, D> &v1, const Vec<T, D> &v2) {T s = 0; for (int i=0; i<D; i++) s += sqr(v1[i] - v2[i]); return s;} // out of range risk for T = byte, ...
template<class T, int D> inline T vecDist(const Vec<T, D> &v1, const Vec<T, D> &v2) { return sqrt(vecSqrDist(v1, v2)); } // out of range risk for T = byte, ...

inline Rect Vec4i2Rect(Vec4i &v){return Rect(Point(v[0] - 1, v[1] - 1), Point(v[2], v[3])); }

Point const DIRECTION4[5] = {
	Point(1, 0),  //Direction 0: right
	Point(0, 1),  //Direction 1: bottom
	Point(-1, 0), //Direction 2: left
	Point(0, -1), //Direction 3: up
	Point(0, 0),
};  //format: {dx, dy}


#include "CmFile.h"
#include "NVTimer.h"
#include "ValStructVec.h"


