#ifndef MIEPY_CUDA_H
#define MIEPY_CUDA_H

#import <torch/extension.h>

void Mie_cal(double m_r, double m_i, torch::Tensor r, int num_r, torch::Tensor wavelength,
             torch::Tensor Qback3, torch::Tensor Qext3);

#endif
