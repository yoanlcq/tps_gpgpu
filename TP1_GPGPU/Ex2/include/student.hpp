/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.hpp
* Author: Maxime MARIA
*/


#ifndef __STUDENT_HPP
#define __STUDENT_HPP

#include <vector>

#include "common.hpp"

namespace IMAC
{
    // Kernel: computes dev_res = dev_a + dev_b
    // - size: arrays size
    // - dev_a, dev_b: input arrays
    // - dev_res: output array
    __global__ void sumArraysCUDA(const int size, const int *const dev_a, const int *const dev_b, int *const dev_res);
    
    // - size: arrays size
    // - a, b: input arrays (on host)
    // - res: output array (on host) (allocated)
    void studentJob(const int size, const int *const a, const int *const b, int *const res);
}

#endif
