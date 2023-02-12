#ifndef MIEPY_KERNEL_H
#define MIEPY_KERNEL_H

#include <THC/THC.h>

void Mie_cal_Laucher(double m_r, double m_i, double *r, int num_r, double *wavelength,
                     double *Qback3, double *Qext3,
                     cudaStream_t stream);



#endif
