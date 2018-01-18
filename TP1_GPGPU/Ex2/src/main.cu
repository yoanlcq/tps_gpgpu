/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
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

namespace IMAC
{
	const int DEFAULT_VECTOR_SIZE = 256;

	// Print program usage
	void printUsageAndExit(const char *prg) 
	{
		std::cerr	<< "Usage: " << prg << std::endl
					<< " \t -n <N>: <N> is the size of the vectors (default is " 
					<< DEFAULT_VECTOR_SIZE << ")" << std::endl << std::endl;
		exit(EXIT_FAILURE);
	}

	void sumArraysCPU(const int n, const int *const a, const int *const b, int *const c)
	{
		std::cout << "Addition on CPU (sequential)"	<< std::endl;
		ChronoCPU chrCPU;
		chrCPU.start();
		for (int i = 0; i < n; ++i)
		{
			c[i] = a[i] + b[i];
		} 
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
	}

	// Compate two arrays (a and b) of size n. Return true if equal
	bool compare(const int *const a, const int *const b, const int n)
	{
		for (int i = 0; i < n; ++i)
		{
			if (a[i] != b[i])
			{
				std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << std::endl;
				return false; 
			}
		}
		return true;
	}

	// Main function
	void main(int argc, char **argv) 
	{	
		int N = DEFAULT_VECTOR_SIZE;

		// Parse command line
		for (int i = 1; i < argc; ++i ) 
		{
			if (!strcmp( argv[i], "-n")) // Arrays size
			{
				if (sscanf(argv[++i], "%d", &N) != 1)
				{
					printUsageAndExit(argv[0]);
				}
			}
			else
			{
				printUsageAndExit(argv[0]);
			}
		}

		std::cout << "Summing vectors of size " << N << std::endl << std::endl;

		ChronoCPU chrCPU;

		// Allocate arrays on CPU
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( N * 3 * sizeof( int ) ) >> 20 ) << " MB on Host" << std::endl;
		chrCPU.start();
		int *a = new int[N];
		int *b = new int[N];
		int *resCPU = new int[N];
		chrCPU.stop();
		std::cout 	<< " -> Done : " << chrCPU.elapsedTime() << " ms" << std::endl << std::endl;
		
		// Init arrays 
		for (int i = 0; i < N; ++i) 
		{
		 	a[i] = i;
		 	b[i] = -2 * i;
		}

		// Computation on CPU
		sumArraysCPU(N, a, b, resCPU);
		
		std::cout 	<< "============================================"	<< std::endl
					<< "              STUDENT'S JOB !               "	<< std::endl
					<< "============================================"	<< std::endl;

		// Allocate array to retrieve result from device
		int *resGPU = new int[N];
		// Call student's code
		studentJob(N, a, b, resGPU);
		
		std::cout << "============================================"	<< std::endl << std::endl;

		std::cout << "Checking result..." << std::endl;
		if (compare(resCPU, resGPU, N))
		{
			std::cout << " -> Well done!" << std::endl;
		}
		else
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}

		// Free memory
		delete[] a;
		delete[] b;
		delete[] resCPU;
		delete[] resGPU;

		// // 3 arrays for CUDA
		// int *dev_a, *dev_b, *dev_res;

		// cout << "Allocating " << ( ( N * 3 * sizeof( int ) ) >> 20 ) << " MB on Device" << endl;
		// ChronoGPU chrGPU;
		// chrGPU.start();

		// allocateArraysCUDA( N, &dev_a, &dev_b, &dev_res );

		// chrGPU.stop();
		// const float timeAllocGPU = chrGPU.elapsedTime();
		// cout << "-> Done : " << std::fixed << std::setprecision(2) << timeAllocGPU << " ms" << endl << endl;

		// cout << "Copying data from Host to Device" << std::endl;

		// chrGPU.start();

		// copyFromHostToDevice( N, a, b, &dev_a, &dev_b );

		// chrGPU.stop();
		// const float timeHtoDGPU = chrGPU.elapsedTime();
		// cout << "-> Done : " << timeHtoDGPU << " ms" << endl << endl;

		// // Free useless memory on CPU
		// delete[] a;
		// delete[] b;

		// cout << "Summming vectors" << endl;

		// chrGPU.start();

		// launchKernel( N, dev_a, dev_b, dev_res ); 

		// chrGPU.stop();
		// const float timeComputeGPU = chrGPU.elapsedTime();
		// cout << "-> Done : " << std::fixed << std::setprecision(2) << timeComputeGPU << " ms" << endl << endl;
			
		// cout << "Copying data from Device to Host" << std::endl;

		// int *resGPU = new int[N];

		// chrGPU.start();

		// copyFromDeviceToHost( N, resGPU, dev_res );

		// chrGPU.stop();
		// const float timeDtoHGPU = chrGPU.elapsedTime();
		// cout << "-> Done : " << std::fixed << std::setprecision(2) << timeDtoHGPU << " ms" << endl << endl;

		// freeArraysCUDA( dev_a, dev_b, dev_res );

		// cout << "============================================"	<< endl;
		// cout << "              Checking results              "	<< endl;
		// cout << "============================================"	<< endl;

		// for ( int i = 0; i < N; ++i ) {
		// 	if ( resCPU[i] != resGPU[i] ) {
		// 		cerr << "Error at index " << i << " CPU:  " << resCPU[i] << " - GPU: " << resGPU[i] <<" !!!" << endl;
		// 		cerr << "Retry!" << endl << endl;
		// 		delete resCPU;
		// 		delete resGPU;
		// 		exit( EXIT_FAILURE );
		// 	}
		// }

		// delete resCPU;
		// delete resGPU;

		// cout << "Congratulations! Job's done!" << endl << endl;

		// cout	<< "============================================"			<< endl;
		// cout	<< "            Times recapitulation            "			<< endl;
		// cout	<< "============================================"			<< endl;
		// cout	<< "-> CPU	Sequential"										<< endl;
		// cout	<< "   - Allocation:     "	<< std::fixed << std::setprecision(2) 
		// 									<< timeAllocCPU << " ms "		<< endl;
		// cout	<< "   - Computation:    "	<< std::fixed << std::setprecision(2) 
		// 									<< timeComputeCPU << " ms"		<< endl;
		// cout	<< "-> CPU	Sequential"										<< endl;
		// cout	<< "   - Computation:    "	<< std::fixed << std::setprecision(2) 
		// 									<< timeComputeCPUOMP << " ms"	<< endl;
		// cout	<< "-> GPU	"												<< endl;
		// cout	<< "   - Allocation:     "	<< std::fixed << std::setprecision(2) 
		// 									<< timeAllocGPU << " ms "		<< endl;
		// cout	<< "   - Host to Device: "	<< std::fixed << std::setprecision(2) 
		// 									<< timeHtoDGPU << " ms"			<< endl;
		// cout	<< "   - Computation:    "	<< std::fixed << std::setprecision(2) 
		// 									<< timeComputeGPU << " ms"		<< endl;
		// cout	<< "   - Device to Host: "	<< std::fixed << std::setprecision(2) 
		// 									<< timeDtoHGPU << " ms "		<< endl 
		// 		<< endl;

		// return EXIT_SUCCESS;
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
