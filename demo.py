import torch
import MiePy

def Mie_cal_radii_v4(m_r:float, m_i:float, r:torch.Tensor):
    """
        返回类型是torch.double
    """
    d_r = r.cuda().to(torch.float64)
    num_r = len(d_r)
    d_wavelength = torch.Tensor([355, 532, 1064]).cuda().to(torch.float64)
    d_Qback3 = torch.zeros(3*num_r).cuda().to(torch.float64)
    d_Qext3 = torch.zeros(3*num_r).cuda().to(torch.float64)
    MiePy.calc(m_r, m_i, d_r, num_r, d_wavelength, d_Qback3, d_Qext3)
    d_Qback3 = d_Qback3.view(3, -1).T
    d_Qext3 = d_Qext3.view(3, -1).T
    return d_Qback3.cpu(), d_Qext3.cpu()

if __name__ == '__main__':
    bak ,ext = Mie_cal_radii_v4(1.37, 0.0, torch.arange(1, 10000+1))
    print(bak[1000], ext[1000])
    print(bak[9000], ext[9000])
    print(bak[-1], ext[-1])

