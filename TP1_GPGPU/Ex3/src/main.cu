/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: main.cpp
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

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -f <F>: <F> image file name" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	// Computes sepia of 'input' and stores result in 'output'
	void sepiaCPU(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for (uint i = 0; i < height; ++i) 
		{
			for (uint j = 0; j < width; ++j) 
			{
				const uint id = (i * width + j) * 3;
				const uchar inR = input[id];
				const uchar inG = input[id + 1];
				const uchar inB = input[id + 2];
				output[id] = static_cast<uchar>( std::min<float>( 255.f, ( inR * .393f + inG * .769f + inB * .189f ) ) );
				output[id + 1] = static_cast<uchar>( std::min<float>( 255.f, ( inR * .349f + inG * .686f + inB * .168f ) ) );
				output[id + 2] = static_cast<uchar>( std::min<float>( 255.f, ( inR * .272f + inG * .534f + inB * .131f ) ) );
			}
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Compare two vectors
	bool compare(const std::vector<uchar> &a, const std::vector<uchar> &b)
	{
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			return false;
		}
		for (uint i = 0; i < a.size(); ++i)
		{
			// Floating precision can cause small difference between host and device
			if (std::abs(a[i] - b[i]) > 1)
			{
				std::cout << "Error at index " << i << ": a = " << uint(a[i]) << " - b = " << uint(b[i]) << std::endl;
				return false; 
			}
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		char fileName[2048];

		// Parse command line
		if (argc == 1) 
		{
			std::cerr << "Please give a file..." << std::endl;
			printUsageAndExit(argv[0]);
		}

		for (int i = 1; i < argc; ++i) 
		{
			if (!strcmp(argv[i], "-f")) 
			{
				if (sscanf(argv[++i], "%s", fileName) != 1)
				{
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}
		
		// Get input image
		std::vector<uchar> input;
		uint width;
		uint height;

		std::cout << "Loading " << fileName << std::endl;
		unsigned error = lodepng::decode(input, width, height, fileName, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		std::cout << "Image has " << width << " x " << height << " pixels (RGBA)" << std::endl;

		// Create 2 output images
		std::vector<uchar> outputCPU(3 * width * height);
		std::vector<uchar> outputGPU(3 * width * height);

		// Prepare output file name
		const std::string fileNameStr(fileName);
		std::size_t lastPoint = fileNameStr.find_last_of(".");
		std::string ext = fileNameStr.substr(lastPoint);
		std::string name = fileNameStr.substr(0,lastPoint);
		std::string outputCPUName = name + "_SepiaCPU" + ext;
		std::string outputGPUName = name + "_SepiaGPU" + ext;

		// Computation on CPU
		sepiaCPU(input, width, height, outputCPU);
		
		std::cout << "Save image as: " << outputCPUName << std::endl;
		error = lodepng::encode(outputCPUName, outputCPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::encode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		studentJob(input, width, height, outputGPU);

		std::cout << "Save image as: " << outputGPUName << std::endl;
		error = lodepng::encode(outputGPUName, outputGPU, width, height, LCT_RGB);
		if (error)
		{
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		
		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(outputCPU, outputGPU))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
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
