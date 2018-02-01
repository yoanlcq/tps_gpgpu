/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: conv_utils.hpp
* Author: Maxime MARIA
*/

#ifndef __CONV_UTILS_HPP
#define __CONV_UTILS_HPP

#include <string>
#include <vector>

namespace IMAC
{
	enum 
	{
		BUMP_3x3 = 0,
		SHARPEN_5x5, 
		EDGE_DETECTION_7x7, 
		MOTION_BLUR_15x15, 
	};

	std::string convertConvTypeToString(const unsigned int convType);

    void initConvolutionMatrix(const unsigned int convType, std::vector<float> &matConv, unsigned int &matSize);
}

#endif
