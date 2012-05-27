/*
 * ANExporter.h
 *
 *  Created on: 09.02.2011
 *      Author: dgrat
 */

#ifndef ANEXPORTER_H_
#define ANEXPORTER_H_

#include <string>

namespace ANN {

class AbsNet;
class F2DArray;
class F3DArray;


/**
 * \brief Exports the net to a format usable with CUDA or OpenCL and hosts functionality to save it to the filesystem.
 *
 * @author Daniel "dgrat" Frenzel
 */
class Exporter {
private:
	AbsNet *m_pNet;
	F2DArray 	*m_pNeurons;
	F2DArray 	*m_pDeltas;
	F3DArray 	*m_pWeights;

public:
	Exporter(AbsNet *pNet);
	virtual ~Exporter();

	/**
	 * Saves the weights of a network to the filesystem
	 * This function uses bzip2 compression (big nets are very memory consuming).
	 * @param sPath Path of the filesystem with name of the new file.
	 */
	void ExpToFS(std::string sPath) const; 	// TODO further testing for this function required

	/**
	 * Exports the weights of the net to a "normal" 1D array.
	 * Necessary for processing data with CUDA.
	 */
//	F3DArray 	*ExpToWArray(); 	// TODO further testing for this function required
	/**
	 * Exports the Neurons of the net to a "normal" 1D array.
	 * Necessary for processing data with CUDA.
	 */
//	F2DArray 	*ExpToNArray(); 	// TODO further testing for this function required
	/**
	 * Exports the error deltas of the net to a "normal" 1D array.
	 * Necessary for processing data with CUDA.
	 */
//	F2DArray 	*ExpToDArray(); 	// TODO further testing for this function required
};

}
#endif /* ANEXPORTER_H_ */
