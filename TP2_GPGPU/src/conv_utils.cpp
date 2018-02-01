/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: conv_utils.cpp
* Author: Maxime MARIA
*/

#include "conv_utils.hpp"

#include <exception>   
#include <stdexcept>

namespace IMAC
{
	void initConvolutionMatrix(const unsigned int convType, std::vector<float> &matConv, unsigned int &matSize)
	{
		switch ( convType ) 
		{
		case BUMP_3x3 :
			{
			matSize = 3;
			matConv.resize(matSize * matSize);
			matConv[0] = -2.f;	matConv[1] = -1.f;	matConv[2] = 0.f;
			matConv[3] = -1.f;	matConv[4] = 1.f;	matConv[5] = 1.f;
			matConv[6] = 0.f;	matConv[7] = 1.f;	matConv[8] = 2.f;
			}
			break;
		case SHARPEN_5x5 :
			{
			matSize = 5;
			matConv.resize(matSize * matSize);
			matConv[0] = -0.125f;	matConv[1] = -0.125f;	matConv[2] = -0.125f;	matConv[3] = -0.125f;	matConv[4] = -0.125f;	
			matConv[5] = -0.125f;	matConv[6] = 0.25f;		matConv[7] = 0.25f;		matConv[8] = 0.25f;		matConv[9] = -0.125f;		
			matConv[10] = -0.125f;	matConv[11] = 0.25f;	matConv[12] = 1.f;		matConv[13] = 0.25f;	matConv[14] = -0.125f;	
			matConv[15] = -0.125f;	matConv[16] = 0.25f;	matConv[17] = 0.25f;	matConv[18] = 0.25f;	matConv[19] = -0.125f;		
			matConv[20] = -0.125f;	matConv[21] = -0.125f;	matConv[22] = -0.125f;	matConv[23] = -0.125f;	matConv[24] = -0.125f;;
			}
			break;	
		case EDGE_DETECTION_7x7 :
			{
			matSize = 7;
			matConv.resize(matSize * matSize);			
			matConv[0] = 0.1f;	matConv[1] = 0.2f;	matConv[2] = 0.5f;		matConv[3] = 0.8f;		matConv[4] = 0.5f;		matConv[5] = 0.2f;	matConv[6] = 0.1f;	
			matConv[7] = 0.2f;	matConv[8] = 1.1f;	matConv[9] = 2.5f;		matConv[10] = 2.7f;		matConv[11] = 2.5f;		matConv[12] = 1.1f;	matConv[13] = 0.2f;	
			matConv[14] = 0.5f;	matConv[15] = 2.5f;	matConv[16] = 0.f;		matConv[17] = -6.1f;	matConv[18] = 0.f;		matConv[19] = 2.5f;	matConv[20] = 0.5f;	
			matConv[21] = 0.8f;	matConv[22] = 2.7f;	matConv[23] = -6.1f;	matConv[24] = -20.f;	matConv[25] = -6.1f;	matConv[26] = 2.7f;	matConv[27] = 0.8f;
			matConv[28] = 0.5f;	matConv[29] = 2.5f;	matConv[30] = 0.f;		matConv[31] = -6.1f;	matConv[32] = 0.f;		matConv[33] = 2.5f;	matConv[34] = 0.5f;
			matConv[35] = 0.2f;	matConv[36] = 1.1f;	matConv[37] = 2.5f;		matConv[38] = 2.7f;		matConv[39] = 2.5f;		matConv[40] = 1.1f;	matConv[41] = 0.2f;
			matConv[42] = 0.1f;	matConv[43] = 0.2f;	matConv[44] = 0.5f;		matConv[45] = 0.8f;		matConv[46] = 0.5f;		matConv[47] = 0.2f;	matConv[48] = 0.1f;
			}
			break;
		case MOTION_BLUR_15x15:
			{
			matSize = 15;
			matConv.resize(matSize * matSize);
			for ( unsigned int j = 0; j < matSize; ++j ) 
			{
				for ( unsigned int i = 0; i < matSize; ++i ) 
				{
					const unsigned int id = j * matSize + i;
					matConv[id] = i == j ? 1.f/9.f : 0.f;
				}
			}
			}
			break;
		default:
			throw std::runtime_error("Error initializing convolution matrix, wrong type");
		}
	}
	
	std::string convertConvTypeToString(const unsigned int convType)
	{
		std::string convStr;
		switch ( convType ) 
		{
		case BUMP_3x3 :
			convStr = "_Bump_3x3";
			break;
		case SHARPEN_5x5 :
			convStr = "_Sharpen_5x5";
			break;	
		case EDGE_DETECTION_7x7 :
			convStr = "_EdgeDetection_7x7";
			break;
		case MOTION_BLUR_15x15:
			convStr = "_MotionBlur_15x15";
			break;
		default:
			throw std::runtime_error("Error getting convolution name, wrong type");
		}
		return convStr;
	}
}
