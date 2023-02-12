#include "MiePy_kernel.h"
#include <torch/extension.h>
#include <THC/THC.h>

bool Mie_cal(double m_r, double m_i, torch::Tensor r, int num_r, torch::Tensor wavelength,
             torch::Tensor Qback3, torch::Tensor Qext3){
    // Grab the input tensor
    double* r_flat = r.data_ptr<double>();
    double* wavelength_flat = wavelength.data_ptr<double>();
    double* Qback3_flat = Qback3.data_ptr<double>();
    double* Qext3_flat = Qext3.data_ptr<double>();

    Mie_cal_Laucher(m_r, m_i, r_flat, num_r, wavelength_flat,
                    Qback3_flat, Qext3_flat,
                    at::cuda::getCurrentCUDAStream());

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("calc", &Mie_cal, "Mie scatter Calculation with CUDA");
}
