/*********************************************************************************/
/* This source code is free for academic use.                                    */
/* If you find BING useful in your research, please consider citing our paper.   */
/*********************************************************************************/

#include "stdafx.h"
#include "BING.h"
#include "ValStructVec.h"


// Uncomment line line 19 in BING.cpp to remove counting times of image reading.

void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz, CStr &dataPath);

int main(int argc, char* argv[])
{
	if(argc < 2){
		std::cerr << "Please pass the data path to as first argument" << std::endl;
		return 1;
	}
	CStr dataPath = argv[1];
	RunObjectness("WinRecall.m", 2, 8, 2, 130, dataPath);
	return 0;
}

void RunObjectness(CStr &resName, double base, int W, int NSS, int numPerSz, CStr &dataPath)
{
	srand(131);
	DataSetVOC voc2007(dataPath); 
	voc2007.loadAnnotations();

	cout << "Dataset:'" << _S(voc2007.wkDir) << "' with " << voc2007.trainNum << " training and " << voc2007.testNum << " testing" << endl;
	cout << _S(resName) << " Base = " << base << ", W = " << W << ", NSS = " << NSS << ", perSz = " << numPerSz << endl;

	Objectness objNess(voc2007, base, W, NSS);
    
	vector<vector<Vec4i>> boxes;
	//objNess.getObjBndBoxesForTests(boxes, 250);
	objNess.getObjBndBoxesForTestFast(boxes, numPerSz);
	//objNess.getRandomBoxes(boxes);
	//objNess.evaluatePerClassRecall(boxes, resName, 2000);
	//objNess.illuTestReults(boxes);
}

