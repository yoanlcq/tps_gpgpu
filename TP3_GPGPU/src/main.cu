/*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: main.cu
* Author: Maxime MARIA
*/

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <ctime>
#include <climits>
#include <exception>

#include "student.hpp"
#include "chronoCPU.hpp"

namespace IMAC
{
	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -n <N>: size of vector is <N> (default is 2^23)" << std::endl 
					<< " \t -2n <N>: size of vector is 2^<N> (default N = 23, then size is 2^23)" << std::endl 
					<< std::endl;
		exit(EXIT_FAILURE);
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		uint power = 23;
		uint size = 1 << power;

		// Parse command line
		for ( int i = 1; i < argc; ++i ) 
		{
			if ( !strcmp( argv[i], "-2n" ) ) 
			{
				if ( sscanf( argv[++i], "%u", &power ) != 1 )
				{
					std::cerr << "No power provided..." << std::endl;
					printUsageAndExit( argv[0] );
				}
				else
				{
					size = 1 << power;
				}
			}
			else if ( !strcmp( argv[i], "-n" ) ) 
			{
				if ( sscanf( argv[++i], "%u", &size ) != 1 )
				{
					std::cerr << "No size provided..." << std::endl;
					printUsageAndExit( argv[0] );
				}
			}
			else
			{
				std::cerr << "Unrognizeed argument: " << argv[i] << std::endl;
				printUsageAndExit( argv[0] );
			}
		}
		
		std::cout << "Max reduce for an array of size " << size << std::endl;
		
		std::cout 	<< "Allocating array on host, " 
					<< ( (size * sizeof(uint)) >> 20 ) << " MB" << std::endl;
		std::vector<uint> array(size, 0);

		std::cout << "Initiliazing array..." << std::endl; // ;-)
		std::srand(std::time(NULL));
		const uint maxRnd = 79797979;
		for (uint i = 0; i < size; ++i)
		{
			if(i % 32 == 0)
				array[i] = std::rand() % maxRnd;
			else
				array[i] = 79;
		} 
		if (std::rand() % 2 == 0 ) array[size - 1] = maxRnd + std::rand() % 797979;

		// Find max on CPU
		uint resultCPU = 0;

		ChronoCPU chrCPU;
		
		std::cout << "Process on CPU (sequential)"	<< std::endl;
		chrCPU.start();
		for (uint i = 0; i < size; ++i)
		{
			resultCPU = std::max<uint>(resultCPU,array[i]);
		}
		chrCPU.stop();
		std::cout 	<< " -> Done : max is " << resultCPU << " (" << chrCPU.elapsedTime() << " ms)" << std::endl << std::endl;

		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;
		try
		{
			studentJob(array, resultCPU);
		}
		catch(const std::exception &e)
		{
			throw;
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
