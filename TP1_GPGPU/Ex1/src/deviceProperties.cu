/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 1 : Vérifions le materiel
*
* File: deviceProperties.cu
* Author: Maxime MARIA
*/

#include "deviceProperties.hpp"
#include "common.hpp"

using namespace std;

int countDevices() 
{
	int cptDevice = 0;
	HANDLE_ERROR( cudaGetDeviceCount( &cptDevice ) );
	return cptDevice;
}

void printDeviceProperties( const int i ) 
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
		
	cout << "============================================"	<< endl;
	cout << "               Device " << i					<< endl;
	cout << "============================================"	<< endl;
	cout << " - name: "					<< prop.name							<< endl;
	cout << " - pciBusId: "				<< prop.pciBusID						<< endl;
	cout << " - pciDeviceId: "			<< prop.pciDeviceID						<< endl;
	cout << " - pciDomainId: "			<< prop.pciDomainID						<< endl;
    
	cout << "============================================" << endl;
	cout << " - cuda version: "					<< prop.major << "." << prop.minor					<< endl;
	cout << " - is integrated: "				<< ( prop.integrated ? "yes" : "no" )				<< endl;
	cout << " - max kernel execution time: "	<< ( prop.kernelExecTimeoutEnabled ? "yes" : "no" )	<< endl;
	cout << " - device can overlap: "			<< ( prop.deviceOverlap ?	"yes" : "no" )			<< endl;
	cout << " - concurrent kernels allowed: "	<< ( prop.concurrentKernels ? "yes" : "no" )		<< endl;
	cout << " - compute mode: "					<< ( prop.computeMode ? "yes" : "no" )				<< endl;
	
	cout << "============================================" << endl;
	cout << " - total Mem: "				<< ( prop.totalGlobalMem >> 20 ) << " Mo"		<< endl;
	cout << " - shared Mem: "				<< ( prop.sharedMemPerBlock >> 10 ) << " Ko"	<< endl;
	cout << " - total constant memory: "	<< ( prop.totalConstMem >> 10 ) << " Ko"		<< endl;
	cout << " - memory pitch: "				<< ( prop.memPitch >> 20 ) << " Mo"				<< endl;
	cout << " - can map host memory: "		<< ( prop.canMapHostMemory ? "yes" : "no" )		<< endl;
	cout << " - memory bus width: "			<< prop.memoryBusWidth	<< "-bit"				<< endl;
	cout << " - memory clock rate: "		<< prop.memoryClockRate	* 1e-3f << " MHz"		<< endl;
	cout << " - unified addressing: "		<< ( prop.unifiedAddressing ? "yes" : "no" )	<< endl;

	cout << "============================================" << endl;
	cout << " - registers per blocks: "				<< prop.regsPerBlock				<< endl;
	cout << " - warpSize: "							<< prop.warpSize					<< endl;
	cout << " - max threads dim: "					<< prop.maxThreadsDim[0] << ", "
													<< prop.maxThreadsDim[1] << ", "
													<< prop.maxThreadsDim[2]			<< endl;
	cout << " - max threads per block: "			<< prop.maxThreadsPerBlock			<< endl;
	cout << " - max threads per multiprocessor: "	<< prop.maxThreadsPerMultiProcessor	<< endl;
	cout << " - max grid size: "					<< prop.maxGridSize[0] << ", "
													<< prop.maxGridSize[1] << ", "
													<< prop.maxGridSize[2]				<< endl;
	cout << " - multiProcessor count: "				<< prop.multiProcessorCount			<< endl;

	cout << "============================================" << endl;
	cout << " - texture alignment: "	<< prop.textureAlignment		<<endl;
	cout << " - max Texture 1D: "		<< prop.maxTexture1D			<<endl;
	cout << " - max texture 2d: "		<< prop.maxTexture2D[0] << ", " 
										<< prop.maxTexture2D[1]			<< endl;
	cout << " - max texture 3D: "		<< prop.maxTexture3D[0] << ", "
										<< prop.maxTexture3D[1] << ", "
										<< prop.maxTexture3D[2]			<< endl;

	cout << "============================================" << endl;
	cout << " - number of asynchronous engines: "	<< prop.asyncEngineCount				<< endl;
	cout << " - clock frequency: "					<< prop.clockRate * 1e-3f << " MHz"		<< endl;
	cout << " - ECCEnabled: "						<< ( prop.ECCEnabled ? "yes" : "no"	)	<< endl;
	cout << " - level 2 cache size: "				<< ( prop.l2CacheSize >> 10 ) << " Ko"	<< endl << endl;
}
