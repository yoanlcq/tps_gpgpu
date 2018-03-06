/*
* File: chronoGPU.cu
* Author: Maxime MARIA
*/

#include <ChronoGPU.hpp>
#include <handle_cuda_error.hpp>
#include <iostream>

ChronoGPU::ChronoGPU() 
	: m_started( false ) {
	handle_cuda_error( cudaEventCreate( &m_start ) );
	handle_cuda_error( cudaEventCreate( &m_end ) );
}

ChronoGPU::~ChronoGPU() {
	if ( m_started ) {
		stop();
		std::cerr << "ChronoGPU::~ChronoGPU(): chrono wasn't turned off!" << std::endl; 
	}
	handle_cuda_error( cudaEventDestroy( m_start ) );
	handle_cuda_error( cudaEventDestroy( m_end ) );
}

void ChronoGPU::start() {
	if ( !m_started ) {
		handle_cuda_error( cudaEventRecord( m_start, 0 ) );
		m_started = true;
	}
	else
		std::cerr << "ChronoGPU::start(): chrono is already started!" << std::endl;
}

void ChronoGPU::stop() {
	if ( m_started ) {
		handle_cuda_error( cudaEventRecord( m_end, 0 ) );
		handle_cuda_error( cudaEventSynchronize( m_end ) );
		m_started = false;
	}
	else
		std::cerr << "ChronoGPU::stop(): chrono wasn't started!" << std::endl;
}

float ChronoGPU::elapsedTime() { 
	float time = 0.f;
	handle_cuda_error( cudaEventElapsedTime( &time, m_start, m_end ) );
	return time;
}
