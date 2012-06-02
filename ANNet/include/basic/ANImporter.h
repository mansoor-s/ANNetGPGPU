/*
 * ANImporter.h
 *
 *  Created on: 10.02.2011
 *      Author: dgrat
 */

#ifndef ANIMPORTER_H_
#define ANIMPORTER_H_

#include <string>

namespace ANN {

class ConTable;
class AbsNet;


/**
 * \brief Imports the net from filesystem.
 *
 * @author Daniel "dgrat" Frenzel
 */
class Importer {
public:
	AbsNet *m_pNet;

	Importer(AbsNet *pNet);
	virtual ~Importer();

	/**
	 * Loads the weights of a network from the filesystem
	 * @param sPath Path of the filesystem with name of the file inheriting the weight values.
	 */
	void ImpFromFS(std::string sPath);

	/**
	 * Functions uses the data from LoadNetwork() and creates the net.
	 * I. e. has to create all objects (layers, neurons, edges).
	 */
	void CreateNet(const ConTable &);
};

}
#endif /* ANIMPORTER_H_ */
