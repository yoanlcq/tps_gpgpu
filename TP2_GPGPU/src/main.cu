/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: main.cu
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>

#include "student.hpp"
#include "chronoCPU.hpp"
#include "lodepng.h"
#include "conv_utils.hpp"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name (required)" << std::endl
					<< " \t -c <C>: <C> convolution type (required)" << std::endl 
					<< " \t --- " << BUMP_3x3 << " = Bump 3x3" << std::endl
					<< " \t --- " << SHARPEN_5x5 << " = Sharpen 5x5" << std::endl
					<< " \t --- " << EDGE_DETECTION_7x7 << " = Edge detection 7x7" << std::endl
					<< " \t --- " << MOTION_BLUR_15x15 << " = Motion Blur 15x15" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	float clampf(const float val, const float min , const float max) 
	{
		return std::min<float>(max, std::max<float>(min, val));
	}
	
	void convCPU(	const std::vector<uchar4> &input, const uint imgWidth, const uint imgHeight, 
					const std::vector<float> &matConv, const uint matSize, 
					std::vector<uchar4> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for ( uint y = 0; y < imgHeight; ++y )
		{
			for ( uint x = 0; x < imgWidth; ++x ) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)input[idPixel].x * matConv[idMat];
						sum.y += (float)input[idPixel].y * matConv[idMat];
						sum.z += (float)input[idPixel].z * matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)clampf( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clampf( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clampf( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char fileName[2048];
		uint convType;
		// Parse command line
		if (argc != 5) 
		{
			std::cerr << "Wrong number of argument" << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) 
		{
			if (!strcmp(argv[i], "-f")) 
			{
				if (sscanf(argv[++i], "%s", fileName) != 1)
				{
					std::cerr << "No file provided after -f" << std::endl;
					printUsageAndExit(argv[0]);
				}
			}
			else if(!strcmp(argv[i], "-c"))
			{
				if (sscanf(argv[++i], "%u", &convType) != 1)
				{
					std::cerr << "No index after -c" << std::endl;
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> inputUchar;
		uint imgWidth;
		uint imgHeight;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(inputUchar, imgWidth, imgHeight, fileName, LCT_RGBA);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		// Convert to uchar4 for exercise convenience
		std::vector<uchar4> input;
		input.resize(inputUchar.size() / 4);
		for (uint i = 0; i < input.size(); ++i)
		{
			const uint id = 4 * i;
			input[i].x = inputUchar[id];
			input[i].y = inputUchar[id + 1];
			input[i].z = inputUchar[id + 2];
			input[i].w = inputUchar[id + 3];
		}
		inputUchar.clear();
		std::cout << "Image has " << imgWidth << " x " << imgHeight << " pixels (RGBA)" << std::endl;

		// Init convolution matrix
		std::vector<float> matConv;
		uint matSize;
		initConvolutionMatrix(convType, matConv, matSize);

		// Create 2 output images
		std::vector<uchar4> outputCPU(imgWidth * imgHeight);
		std::vector<uchar4> outputGPU(imgWidth * imgHeight);

		
		std::cout << input.size() << " - " << outputCPU.size() << " - " << outputGPU.size() << std::endl;

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string convStr = convertConvTypeToString(convType);
		std::string outputCPUName = name + convStr + "_CPU" + ext;
		std::string outputGPUName = name + convStr + "_GPU" + ext;

		// Computation on CPU
		convCPU(input, imgWidth, imgHeight, matConv, matSize, outputCPU);
		
		std::cout << "Save image as: " << outputCPUName << std::endl;
		error = lodepng::encode(outputCPUName, reinterpret_cast<uchar *>(outputCPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error)
		{
			throw std::runtime_error("Error loadpng::encode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input, imgWidth, imgHeight, matConv, matSize, outputCPU, outputGPU);

		std::cout << "Save image as: " << outputGPUName << std::endl;
		error = lodepng::encode(outputGPUName, reinterpret_cast<uchar *>(outputGPU.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;
	}
}

int main(int argc, char **argv) 
{
	try
	{
		IMAC::main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
