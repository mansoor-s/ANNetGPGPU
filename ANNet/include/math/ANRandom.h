/*
 * ANRandom.h
 *
 *  Created on: 22.01.2011
 *      Author: dgrat
 */

#ifndef RANDOMIZER_H_
#define RANDOMIZER_H_

#include <cstdlib>
#include <cmath>
#include <ctime>


#ifdef __linux__
	#include <sys/times.h>
	/*
	 * not defined in unix os but windows
	 */
	inline long getTickCount() {
		struct tms tm;
		return times(&tm);
	}
#endif /*__linux__*/

namespace ANN {
/*
 * predeclaration of some functions
 */
inline float RandFloat(float begin, float end);
inline int RandInt(int x,int y);

inline void InitTime();
#define INIT_TIME InitTime();

#ifdef WIN32
	/*
	 * for getTickCount()
	 */
	typedef unsigned long 	DWORD;
	typedef unsigned short 	WORD;
	typedef unsigned int 		UNINT32;

	#include <windows.h>
#endif /*WIN32*/

void InitTime() {
	time_t t;
	time(&t);
	srand((unsigned int)t);
}
/*
 * Returns a random number
 * Call of getTickCount() necessary
 */
float RandFloat(float begin, float end) {
	float temp;
	/* swap low & high around if the user makes no sense */
	if (begin > end) {
		temp = begin;
		begin = end;
		end = temp;
	}

	/* calculate the random number & return it */
	return rand() / (RAND_MAX + 1.f) * (end - begin) + begin;
}

//returns a random integer between x and y
int RandInt(int x,int y) {
	return rand()%(y-x+1)+x;
}

}

#endif /* RANDOMIZER_H_ */
