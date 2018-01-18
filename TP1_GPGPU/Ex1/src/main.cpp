/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 1 : Verifions le matériel
*
* File: main.cpp
* Author: Maxime MARIA
*/

#include <iostream>

#include "deviceProperties.hpp"

int main() 
{
	try
	{
		int cptDevice = countDevices();

		std::cout << "-> " << cptDevice << " CUDA capable device(s)" << std::endl << std::endl;
		
		for (int i = 0; i < cptDevice; ++i)
		{
			printDeviceProperties(i);
		}
	}
	catch(const std::exception &e)
	{
		std::cerr << "Exception caught: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	
	return EXIT_SUCCESS;
}
