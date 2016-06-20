#include "mex.h"
#include "stdafx.h"
#include "ValStructVec.h"
#include "FilterBING.h"

class mstream : public std::streambuf {
public:
protected:
	virtual std::streamsize xsputn(const char *s, std::streamsize n) {
		mexPrintf("%.*s", n, s);
		return n;
	}
	virtual int overflow(int c = EOF) {
		if (c != EOF) {
			mexPrintf("%.1s", &c);
		}
		return 1;
	}
};

const double _base = 2.0;
const double _logBase = log(_base);
const int _W = 8;
const int _NSS = 2;
const int _minT = cvCeil(log(10.) / _logBase);
const int _maxT = cvCeil(log(500.) / _logBase);
const int _numT = _maxT - _minT + 1;

vecI _svmSzIdxs;   // Indexes of active size. It's equal to _svmFilters.size() and _svmReW1f.rows
Mat _svmFilter;    // Filters learned at stage I, each is a _H by _W CV_32F matrix
FilterBING _bingF; // BING filter
Mat _svmReW1f;     // Re-weight parameters learned at stage II.

// Read matrix from binary file
bool matRead(CStr& filename, Mat& _M){
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf, sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		cout << "Invalidate CvMat data file " << _S(filename) << endl;
		return false;
	}
	int headData[3]; // Width, height, type
	if (fread(headData, sizeof(int), 3, f) != 3)
                return false;
	Mat M(headData[1], headData[0], headData[2]);
	if (fread(M.data, sizeof(char), M.step * M.rows, f) != M.step * M.rows)
                return false;
	fclose(f);
	M.copyTo(_M);
	return true;
}

// Return -1, 0, or 1 if partial, none, or all loaded
int loadTrainedModel(string modelName)
{
	CStr s1 = modelName + ".wS1", s2 = modelName + ".wS2", sI = modelName + ".idx";
	Mat filters1f, reW1f, idx1i, show3u;
	if (!matRead(s1, filters1f) || !matRead(sI, idx1i)){
		cout << "Can't load model: " << _S(s1) << " or " << _S(sI) << endl;
		return 0;
	}

	normalize(filters1f, show3u, 1, 255, NORM_MINMAX, CV_8U);
	_bingF.update(filters1f);

	_svmSzIdxs = idx1i;
	CV_Assert(_svmSzIdxs.size() > 1 && filters1f.size() == Size(_W, _W) && filters1f.type() == CV_32F);
	_svmFilter = filters1f;

	if (!matRead(s2, _svmReW1f) || _svmReW1f.size() != Size(2, _svmSzIdxs.size())){
		_svmReW1f = Mat();
		return -1;
	}
	return 1; 
}

static inline int bgrMaxDist(const Vec3b &u, const Vec3b &v) 
{ 
	int b = abs(u[0] - v[0]), g = abs(u[1] - v[1]), r = abs(u[2] - v[2]); 
	b = max(b, g); 
	return max(b, r); 
}

void gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u)
{
	const int H = x1i.rows, W = x1i.cols;
	mag1u.create(H, W, CV_8U);
	for (int r = 0; r < H; r++){
		const int *x = x1i.ptr<int>(r), *y = y1i.ptr<int>(r);
		byte* m = mag1u.ptr<byte>(r);
		for (int c = 0; c < W; c++)
			m[c] = min(x[c] + y[c], 255);
	}
}

void gradientMag(CMat &bgr3u, Mat &mag1u)
{
	const int H = bgr3u.rows, W = bgr3u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = bgrMaxDist(bgr3u.at<Vec3b>(y, 1), bgr3u.at<Vec3b>(y, 0)) * 2;
		Ix.at<int>(y, W - 1) = bgrMaxDist(bgr3u.at<Vec3b>(y, W - 1), bgr3u.at<Vec3b>(y, W - 2)) * 2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = bgrMaxDist(bgr3u.at<Vec3b>(1, x), bgr3u.at<Vec3b>(0, x)) * 2;
		Iy.at<int>(H - 1, x) = bgrMaxDist(bgr3u.at<Vec3b>(H - 1, x), bgr3u.at<Vec3b>(H - 2, x)) * 2;
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++){
		const Vec3b *dataP = bgr3u.ptr<Vec3b>(y);
		for (int x = 2; x < W; x++)
			Ix.at<int>(y, x - 1) = bgrMaxDist(dataP[x - 2], dataP[x]);
	}
	for (int y = 1; y < H - 1; y++){
		const Vec3b *tP = bgr3u.ptr<Vec3b>(y - 1);
		const Vec3b *bP = bgr3u.ptr<Vec3b>(y + 1);
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = bgrMaxDist(tP[x], bP[x]);
	}
	gradientXY(Ix, Iy, mag1u);
}

void nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS, int maxPoint, bool fast)
{
	const int _h = matchCost1f.rows, _w = matchCost1f.cols;
	Mat isMax1u = Mat::ones(_h, _w, CV_8U), costSmooth1f;
	ValStructVec<float, Point> valPnt;
	matchCost.reserve(_h * _w);
	valPnt.reserve(_h * _w);
	if (fast){
		blur(matchCost1f, costSmooth1f, Size(3, 3));
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			const float* ds = costSmooth1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
			if (d[c] >= ds[c])
				valPnt.pushBack(d[c], Point(c, r));
		}
	}
	else{
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				valPnt.pushBack(d[c], Point(c, r));
		}
	}

	valPnt.sort();
	for (int i = 0; i < valPnt.size(); i++){
		Point &pnt = valPnt[i];
		if (isMax1u.at<byte>(pnt)){
			matchCost.pushBack(valPnt(i), pnt);
			for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
				Point neighbor = pnt + Point(dx, dy);
				if (!CHK_IND(neighbor))
					continue;
				isMax1u.at<byte>(neighbor) = false;
			}
		}
		if (matchCost.size() >= maxPoint)
			return;
	}
}

void predictBBoxSI(Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, vecI &sz, int NUM_WIN_PSZ = 100, bool fast = true)
{
	const int numSz = _svmSzIdxs.size();
	const int imgW = img3u.cols, imgH = img3u.rows;

	valBoxes.reserve(10000);
	sz.clear(); sz.reserve(10000);
	for (int ir = numSz - 1; ir >= 0; ir--){
		int r = _svmSzIdxs[ir];
		int height = cvRound(pow(_base, r / _numT + _minT)), width = cvRound(pow(_base, r%_numT + _minT));
		if (height > imgH * _base || width > imgW * _base)
			continue;

		height = min(height, imgH), width = min(width, imgW);
		Mat im3u, matchCost1f, mag1u;
		resize(img3u, im3u, Size(cvRound(_W*imgW*1.0 / width), cvRound(_W*imgH*1.0 / height)));
		double ratioX = width / _W, ratioY = height / _W;

		gradientMag(im3u, mag1u);
		matchCost1f = _bingF.matchTemplate(mag1u);
		ValStructVec<float, Point> matchCost;
		nonMaxSup(matchCost1f, matchCost, _NSS, NUM_WIN_PSZ, fast);

		// Find true locations and match values
		int iMax = min(matchCost.size(), NUM_WIN_PSZ);
		for (int i = 0; i < iMax; i++){
			float mVal = matchCost(i);
			Point pnt = matchCost[i];
			Vec4i box(cvRound(pnt.x * ratioX), cvRound(pnt.y*ratioY));
			box[2] = cvRound(min(box[0] + width, imgW));
			box[3] = cvRound(min(box[1] + height, imgH));
			box[0] ++;
			box[1] ++;
			valBoxes.pushBack(mVal, box);
			sz.push_back(ir);
		}
	}
}

void predictBBoxSII(ValStructVec<float, Vec4i> &valBoxes, const vecI &sz)
{
	int numI = valBoxes.size();
	for (int i = 0; i < numI; i++){
		const float* svmIIw = _svmReW1f.ptr<float>(sz[i]);
		valBoxes(i) = valBoxes(i) * svmIIw[0] + svmIIw[1];
	}
	valBoxes.sort();
}

int minIdx(Mat &input, int x1, int y1, int x2, int y2) {
	int min = 1000000;
	for (int i = y1; i <= y2; i++) {
		int *data = input.ptr<int>(i);
		for (int j = x1; j <= x2; j++) {
			if (data[j] < min)
				min = data[j];
		}
	}
	return min;
}

int maxIdx(Mat &input, int x1, int y1, int x2, int y2) {
	int max = -1;
	for (int i = y1; i <= y2; i++) {
		int *data = input.ptr<int>(i);
		for (int j = x1; j <= x2; j++) {
			if (data[j] > max)
				max = data[j];
		}
	}
	return max;
}

double interUnio(const Vec4i &bb, const Vec4i &bbgt)
{
	int bi[4];
	bi[0] = max(bb[0], bbgt[0]);
	bi[1] = max(bb[1], bbgt[1]);
	bi[2] = min(bb[2], bbgt[2]);
	bi[3] = min(bb[3], bbgt[3]);

	double iw = bi[2] - bi[0] + 1;
	double ih = bi[3] - bi[1] + 1;
	double ov = 0;
	if (iw>0 && ih>0){
		double ua = (bb[2] - bb[0] + 1)*(bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1)*(bbgt[3] - bbgt[1] + 1) - iw*ih;
		ov = iw*ih / ua;
	}
	return ov;
}

bool filtersLoaded() 
{ 
	int n = _svmSzIdxs.size(); 
	return n > 0 && _svmReW1f.size() == Size(2, n) && _svmFilter.size() == Size(_W, _W); 
}

void getObjBndBoxes(Mat &img3u, ValStructVec<float, Vec4i> &valBoxes, int numDetPerSize = 130)
{
	CV_Assert(filtersLoaded());
	vecI sz;
	predictBBoxSI(img3u, valBoxes, sz, numDetPerSize, false);
	predictBBoxSII(valBoxes, sz);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	mstream mout;
	std::streambuf *outbuf = std::cout.rdbuf(&mout);

	if (nrhs != 2)
		mexErrMsgTxt("There must only be one input variable!");

	const mwSize *dims = mxGetDimensions(prhs[0]);
	int mrows = (int)dims[0];
	int mcols = (int)dims[1];
	int mlays = (int)dims[2];
	if (mlays != 3)
		mexErrMsgTxt("The input array must have three channels!");
	uchar *data = (uchar*)mxGetPr(prhs[0]);
	
	Mat img3u(mrows, mcols, CV_8UC3);
	for (int i = 0; i < mrows; i++) {
		Vec3b* img = img3u.ptr<Vec3b>(i);
		for (int j = 0; j < mcols; j++) {
			img[j][2] = data[i + j*mrows];
			img[j][1] = data[i + j*mrows + mrows*mcols];
			img[j][0] = data[i + j*mrows + mrows*mcols * 2];
		}
	}

	char *path_buf = mxArrayToString(prhs[1]);
	int status = loadTrainedModel(path_buf);
	if (status != 1)
		return;
	ValStructVec<float, Vec4i> boxesTests;
	getObjBndBoxes(img3u, boxesTests, 130);

	const int NUM = boxesTests.size();
	mxArray *mxBoxesTests = mxCreateNumericMatrix(NUM, 4, mxINT32_CLASS, mxREAL);
	int *_boxesTests = (int*)mxGetPr(mxBoxesTests);
	for (int i = 0; i < 4; i++)
	    for (int j = 0; j < NUM; j++)
		    _boxesTests[i*NUM + j] = boxesTests[j][i];
        plhs[0] = mxBoxesTests;

	std::cout.rdbuf(outbuf);
}
