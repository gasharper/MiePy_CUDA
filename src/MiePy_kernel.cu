#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include <time.h>

#include "MiePy_kernel.h"
//#define M_PI 3.141592653589793

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}


// 定义复数结构体
struct jw_complex{
	double re,im;
};     


//  复数加减乘除运算
__device__ jw_complex cset(double a,double b)/*:7*/     //复数赋值 
{ 
	jw_complex c;
	c.re=a; 
	c.im=b; 
	return c; 
}

__device__ double cnorm(jw_complex z)/*:17*/    // 取模 
{ 
	return(sqrt(z.re*z.re+z.im*z.im)); 
}

__device__ double csqure(jw_complex z)/*:17*/    // 取模方
{ 
	return(z.re*z.re+z.im*z.im); 
}

__device__ jw_complex cadd(jw_complex z,jw_complex w)/*:26*/   //  复数相加 
{ 
	jw_complex c;  
	c.im=z.im+w.im; 
	c.re=z.re+w.re; 
	return c; 
}

__device__ jw_complex csub(jw_complex z,jw_complex w)/*:28*/     //  复数相减 
{ 
	jw_complex c; 
	c.im=z.im-w.im; 
	c.re=z.re-w.re; 
	return c; 
}


__device__ jw_complex cmul(jw_complex z,jw_complex w)/*:30*/   //  复数相乘 
{ 
	jw_complex c; 
	c.re=z.re*w.re-z.im*w.im; 
	c.im=z.im*w.re+z.re*w.im; 
	return c; 
} 

__device__ jw_complex cdiv(jw_complex z,jw_complex w)/*:32*/     //  复数相除 z/w
{ 
	jw_complex c; 
	double r,denom;  
	if((w.re==0)&&(w.im==0))
	{
		printf("Attempt to divide by 0+0i\n"); 
		//exit(1);
	} 
	if(fabs(w.re)>=fabs(w.im))
	{ 
		r=w.im/w.re; 
		denom=w.re+r*w.im; 
		c.re=(z.re+r*z.im)/denom; 
		c.im=(z.im-r*z.re)/denom; 
	}
	else
	{ 
		r=w.re/w.im; 
		denom=w.im+r*w.re; 
		c.re=(z.re*r+z.im)/denom; 
		c.im=(z.im*r-z.re)/denom; 
	} 
	return c; 
}

__device__ jw_complex cinv(jw_complex w)/*:23*/    //  复数 求 倒数 
{ 
	double r,d;  
	if((w.re==0)&&(w.im==0))
	{
		printf("Attempt to invert 0+0i\n"); 
		//exit(1);
	}
	if(fabs(w.re)>=fabs(w.im))
	{ 
		r=w.im/w.re; 
		d=1/(w.re+r*w.im); 
		return cset(d,-r*d); 
	}  
	r=w.re/w.im; 
	d=1/(w.im+r*w.re); 
	return cset(r*d,-d); 
}


//  matlab 和 C 语言混合编程 实现 MIE散射的快速计算
//  功能类似于  bhmie.h,调用形式相同
__device__ void MieKernel(jw_complex* S1, jw_complex* S2, double* qext, double* qsca, double* qback, double* gsca, double x, double m_r, double m_i, long nang)
{
	// ========================= 注意： ==========================
	// 调用该函数前，需要将S1、S2内容初始化为零！
	// 				需要将qsca和gsca置为零！（qext和qback不用管，是最后赋值的）
	// 如：
	// *qsca = 0.0; *gsca = 0.0;//令其为零，由于后面要用到其迭代。
	// for(size_t i=0; i<2*nang-1; i++){
	// 	S1[i] = cset(0.0, 0.0);
	// 	S2[i] = cset(0.0, 0.0);
	// }
	// ==========================================================

	//   预分配存储空间
	jw_complex m = cset(m_r,m_i);    //  复折射率
	jw_complex y;
	double xstop,ymod,nmx;
	long nstop,nn;
	double p = -1.0;  // 对称

	double delta_angle,theta;
	
	//  预分配存储空间
	double* costheta = (double*)malloc(nang*sizeof(double));
	double* pi_n = (double*)malloc(nang*sizeof(double)); //  计算散射系数 an bn 相关
	double* pi_n0 = (double*)malloc(nang*sizeof(double));
	double* pi_n1 = (double*)malloc(nang*sizeof(double));
	double* tau_n = (double*)malloc(nang*sizeof(double));

	jw_complex an,bn,an1,bn1,an_0,an_1,bn_0,bn_1;

	double psi0,psi1,chi0,chi1,psi,chi;     //  Riccati-Bessel functions with real argument X
	jw_complex xi1,xi;
		
	long n,j,jj,en;
	double en_d,fn;


	for (n=0;n<nang;n++)
	{
		pi_n0[n] = 0.0;
		pi_n1[n] = 1.0;
	}
	y = cmul(m,cset(x,0.0));
	ymod = cnorm(y);
	xstop = x + 4*pow(x,1.0/3.0)+2.0;
	nmx = ymod>xstop? ymod:xstop;
	nmx = nmx + 15.0;
	nn = ceil(nmx);

	nstop = (long)xstop;
	/*%%    Logarithmic derivatives calculated from NMX on down
		% Logarithmic derivative D(J) calculated by downward recurrence
		% beginning with initial value (0.,0.) at J=NMX*/
	//D = new_carray(nn);
	struct jw_complex* D = (jw_complex*)malloc(nn*sizeof(jw_complex));  //根据推算，nn最大可以达到1521?
	if(D == NULL){
		printf("====== malloc D failed! ======\n");
	}

	D[nn-1] = cset(0.0,0.0);
	
	for(n=0;n<nn-1;n++)
	{
		en = nn-n;
		D[nn-n-2] = csub(cdiv(cset(en,0.0),y), cinv(cadd(D[nn-n-1],cdiv(cset(en,0.0),y))));
		//printf("%d is %f + i*%f \n",nn-n-2,D[nn-n-2].re,D[nn-n-2].im);
	}

	psi0 = cos(x);
	psi1 = sin(x);
	chi0 = -sin(x);
	chi1 = cos(x);
	xi1 = cset(psi1,-chi1);

	delta_angle = 0.5*M_PI/(double)(nang-1);

	for (n=0;n<nang;n++)
	{
		theta = (double)n*delta_angle;
		costheta[n] = cos(theta);
	}

	for(n=0;n<nstop;n++)
	{
		en_d = (double)n+1.0;
		fn = (2.0*en_d+1.0)/(en_d*(en_d+1.0));

		psi = (2.*en_d-1.)*psi1/x - psi0;  //  递推公式
		chi = (2.*en_d-1.)*chi1/x - chi0;
		xi = cset(psi,-chi);

		if(n>0)
		{
			an1 = an;
			bn1 = bn;
		}
		
		an_0 = csub(cmul(cadd(cdiv(D[n],m),cset(en_d/x,0.0)),cset(psi,0.0)),cset(psi1,0.0));
		an_1 = csub(cmul(cadd(cdiv(D[n],m),cset(en_d/x,0.0)),xi),xi1);
		an = cdiv(an_0,an_1);
		bn_0 = csub(cmul(cadd(cmul(m,D[n]),cset(en_d/x,0.0)),cset(psi,0.0)),cset(psi1,0.0));
		bn_1 = csub(cmul(cadd(cmul(m,D[n]),cset(en_d/x,0.0)),xi),xi1);
		bn = cdiv(bn_0,bn_1);

		*qsca = *qsca + (2.0*en_d+1.0)*(csqure(an)+csqure(bn));
		*gsca = *gsca + ((2.0*en_d+1.0)/(en_d*(en_d+1.0)))*(an.re* bn.re+an.im*bn.im);
		if(n>0)
		{
			*gsca = *gsca + ((en_d-1.)* (en_d+1.)/en_d)*(an1.re*an.re+an1.im*an.im+bn1.re*bn.re+bn1.im*bn.im);
		}

		for(j=0;j<nang;j++)
		{
			pi_n[j] = pi_n1[j];
			tau_n[j] = en_d*costheta[j]*pi_n[j] - (en_d+1.0)*pi_n0[j];
			S1[j] = cadd(S1[j],cmul(cset(fn,0.0),cadd(cmul(an,cset(pi_n[j],0.0)),cmul(bn,cset(tau_n[j],0.0)))));
			S2[j] = cadd(S2[j],cmul(cset(fn,0.0),cadd(cmul(bn,cset(pi_n[j],0.0)),cmul(an,cset(tau_n[j],0.0)))));
		}

		p = -p;
		for(j=0;j<nang-1;j++)
		{
			jj = 2*nang-j-2;
			S1[jj] = cadd(S1[jj],cmul(cset(fn*p,0.0),csub(cmul(an,cset(pi_n[j],0.0)),cmul(bn,cset(tau_n[j],0.0)))));
			S2[jj] = cadd(S2[jj],cmul(cset(fn*p,0.0),csub(cmul(bn,cset(pi_n[j],0.0)),cmul(an,cset(tau_n[j],0.0)))));
		}

		psi0 = psi1;
		psi1 = psi;
		chi0 = chi1;
		chi1 = chi;
		xi1 = cset(psi1,-chi1);

		for(j=0;j<nang;j++)
		{
			pi_n1[j] = ((2.0*en_d+1.0)*costheta[j]*pi_n[j]- (en_d+1.0)*pi_n0[j])/en_d;
			pi_n0[j] = pi_n[j];
		}		

	}
	*gsca = 2.*(*gsca)/(*qsca);
	*qsca = (2.0/(x*x))*(*qsca);
	*qext = (4.0/ (x*x))* S1[0].re;
	*qback = pow((cnorm(S1[2*nang-2])/x),2)/M_PI;

	//   释放存储空空间
	if(costheta!=NULL)free(costheta);
	if(pi_n!=NULL)free(pi_n);
	if(pi_n0!=NULL)free(pi_n0);
	if(pi_n1!=NULL)free(pi_n1);
	if(tau_n!=NULL)free(tau_n);	
    if(D != NULL)free(D);

}

// 核函数
__global__ void Mie_cal_Kernel(double m_r, double m_i, double *r, int num_r, double *wavelength, double *Qback3, double *Qext3)
{
	// 调用 MieKernel(jw_complex* S1, jw_complex* S2, double* qext, double* qsca, double* qback, double* gsca, double x, double m_r, double m_i, long nang)
	size_t i = blockDim.x * blockIdx.x + threadIdx.x; // r index
    if(i < 3*num_r){
		size_t ri = i%num_r, wi = i/num_r;
		double now_r = r[ri];
		double now_wavelength = wavelength[wi];
		double* pQback = Qback3 + i;
		double* pQext = Qext3 + i;

		long nang=3;
		double qsca=0.0, gsca=0.0;
		double x = 2*M_PI*now_r / now_wavelength;
		long N_anle = 2*nang - 1;     //  总计算角度数量
		struct jw_complex *S1 = (jw_complex*)malloc(N_anle*sizeof(jw_complex));  
		struct jw_complex *S2 = (jw_complex*)malloc(N_anle*sizeof(jw_complex));
		memset(S1, 0, N_anle*sizeof(jw_complex));
		memset(S2, 0, N_anle*sizeof(jw_complex));
		
		// 调用MieKernel前需要置qsca/ gsca/ S1/ S2为零！
		MieKernel(S1, S2, pQext, &qsca, pQback, &gsca, x, m_r, m_i, nang);

		if(S1 != NULL)free(S1);
		if(S2 != NULL)free(S2);
	}
	
}

void Mie_cal_Laucher(/* 输入 */double m_r, double m_i, double *r, int num_r, double *wavelength, /* 输出 */ double *Qback3, double *Qext3, /* stream */cudaStream_t stream){
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024); // 设置96MB的heap

	double *d_Qback3, *d_Qext3;
	size_t size = num_r * sizeof(double);
	cudaMalloc((void**)&d_Qback3, 3*size);
	cudaMalloc((void**)&d_Qext3, 3*size);

	// 拷贝数组到GPU（标量自动转化）
	double *d_r, *d_wavelength;
	cudaMalloc((void**)&d_r, size);
	cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_wavelength, 3*sizeof(double));
	cudaMemcpy(d_wavelength, wavelength, 3*sizeof(double), cudaMemcpyHostToDevice);

	// 执行核函数
	dim3 dimBlock(512);//每个块内的线程最大值为1024
	dim3 dimGrid((3*num_r + dimBlock.x - 1)/dimBlock.x);//gridsize最大65535
	Mie_cal_Kernel<<<dimGrid, dimBlock, 0, stream>>>(m_r, m_i, d_r, num_r, d_wavelength, d_Qback3, d_Qext3);
	//CudaCheckError();
	cudaDeviceSynchronize();

	// 结果拷贝到CPU
	cudaMemcpy(Qback3, d_Qback3, 3*size, cudaMemcpyDeviceToHost);
	cudaMemcpy(Qext3, d_Qext3, 3*size, cudaMemcpyDeviceToHost);

	// 释放divice内存
	cudaFree(d_Qback3);
	cudaFree(d_Qext3);
	cudaFree(d_wavelength);
	cudaFree(d_r);
}

__global__ void Simpson_Kernel(double* d_y, int num_r, int m, double dr, /*输出*/ double *s){
	// 计算simpson积分，y.shape=(num_r, m)->(num_r*m,)，dr是等距自变量间隔
	// so=0;
    // se=0;
    // for(i=1; i<n; i++)
    // {
    //     if(i%2==1)
    //     {
    //         so=so+y[i];
    //     }
    //     else
    //     {
    //         se=se+y[i];
    //     }
 
    // }
	// ans=h/3*(y[0]+y[n]+4*so+2*se);
	size_t n = blockDim.x * blockIdx.x + threadIdx.x; // r index
    if(n < m*num_r){
		size_t i = n/num_r, j = n%num_r;// 第i个积分的第j个元素
		if(j&1){
			so = so + y[i];// TODO
		}else{
			se = se + y[i];
		}
	}
}

void Simpson_Laucher(double *y, int num_r, int m, double h, /* output */ double *s){
	double *d_y, *d_s;
	cudaSafeCall(cudaMalloc((void**)&d_y, num_r*m*sizeof(double)));
	cudaSafeCall(cudaMalloc((void**)&d_s, m*sizeof(double)));
	cudaMemcpy(d_y, y, num_r*m*sizeof(double), cudaMemcpyHostToDevice);
	memset(d_s, 0, m*sizeof(double));

	dim3 dimBlock(512);//每个块内的线程最大值为1024
	dim3 dimGrid((num_r*m + dimBlock.x - 1)/dimBlock.x);//gridsize最大65535
	Simpson_Kernel<<<dimGrid, dimBlock>>>(d_y, num_r, m, h, d_s);
	CudaCheckError();
	cudaDeviceSynchronize();

	cudaMemcpy(s, d_s, m*sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_y);
	cudaFree(d_s);
}
