import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft2, ifft2, fftshift, ifftshift

import llops as yp
import arrayfire as af

from IPython import display
import time
import pickle
from dftregistration import dftregistration


def image_upsampling(I_image, Ic_image, upsamp_factor = 1, bg = 0):
    F = lambda x: ifftshift(fft2(fftshift(x)))
    iF = lambda x: ifftshift(ifft2(fftshift(x)))
    
    Nimg, Ncrop, Mcrop = I_image.shape

    N = Ncrop*upsamp_factor
    M = Mcrop*upsamp_factor

    I_image_up = np.zeros((Nimg,N,M))
    Ic_image_up = np.zeros((Nimg,N,M))
    
    for i in range(0,Nimg):
        I_image_up[i] = abs(iF(np.pad(F(np.maximum(0,I_image[i]-bg)),\
                                  (((N-Ncrop)//2,),((M-Mcrop)//2,)),mode='constant')))
        Ic_image_up[i] = abs(iF(np.pad(F(np.maximum(0,Ic_image[i]-bg)),\
                                   (((N-Ncrop)//2,),((M-Mcrop)//2,)),mode='constant')))

    return I_image_up, Ic_image_up


def display_image_movie(image_stack, frame_num, size, pause_time=0.0001):
    f1,ax = plt.subplots(1,1,figsize=size)
    max_val = np.max(image_stack)

    for i in range(0,frame_num):
        if i != 1:
            ax.cla()
        ax.imshow(image_stack[i],cmap='gray',vmin=0,vmax=max_val)
        display.display(f1)
        display.clear_output(wait=True)
        time.sleep(pause_time)
        
        
        
def image_registration(Ic_image_up,usfac):
    Nimg,_,_ = Ic_image_up.shape
    xshift = np.zeros(Nimg)
    yshift = np.zeros(Nimg)

    for i in range(0,Nimg):
        if i == 0:
            yshift[i] == 0
            xshift[i] == 0
        else:
            output = dftregistration(fft2(Ic_image_up[0]),fft2(Ic_image_up[i]),usfac)
            yshift[i] = output[0]
            xshift[i] = output[1]
            
    return xshift, yshift

def af_pad(image, NN, MM, val):
    N,M = image.shape
    Np = N + 2*NN
    Mp = M + 2*MM
    if image.dtype() == af.Dtype.f32:
        image_pad = af.constant(val,Np,Mp)
    else:
        image_pad = af.constant(val*(1+1j*0),Np,Mp)
    image_pad[NN:NN+N,MM:MM+M] = image
    
    return image_pad



class fSIM_solver:
    
    def __init__(self, I_image_up, xshift, yshift, N_bound_pad, lambda_f, pscrop, upsamp_factor, NA_obj, NAs, itr, sol_back_end):
        
        # Basic parameter 
        self.Nimg, self.N, self.M = I_image_up.shape
        self.N_bound_pad = N_bound_pad
        self.Nc = self.N + 2*N_bound_pad
        self.Mc = self.M + 2*N_bound_pad
        self.ps = pscrop/upsamp_factor
        self.itr = itr
        self.sol_back_end = sol_back_end
        
        # Shift variable
        self.xshift = xshift.copy()
        self.yshift = yshift.copy()        
        self.xshift_max = np.int(np.round(np.max(abs(xshift))))
        self.yshift_max = np.int(np.round(np.max(abs(yshift))))
        
        
        # Frequency grid definition to create TF
        fx_c = np.r_[-self.Mc/2:self.Mc/2]/self.ps/self.Mc
        fy_c = np.r_[-self.Nc/2:self.Nc/2]/self.ps/self.Nc

        fxx_c, fyy_c = np.meshgrid(fx_c,fy_c)

        fxx_c = ifftshift(fxx_c)
        fyy_c = ifftshift(fyy_c)
        
        Npp = self.Nc + 2*self.yshift_max
        Mpp = self.Mc + 2*self.xshift_max


        fxp = np.r_[-Mpp/2:Mpp/2]/self.ps/Mpp
        fyp = np.r_[-Npp/2:Npp/2]/self.ps/Npp

        fxxp, fyyp = np.meshgrid(fxp,fyp)
        
        fxxp = ifftshift(fxxp)
        fyyp = ifftshift(fyyp)
        
        self.fxxp = yp.asarray(fxxp, backend=sol_back_end)
        self.fyyp = yp.asarray(fyyp, backend=sol_back_end)

        
        # Initialization of object and pattern
        self.I_obj = np.pad(np.mean(I_image_up,axis=0),(N_bound_pad,),mode='constant')
        self.I_obj = yp.asarray(self.I_obj, backend=sol_back_end)
        self.I_p_whole = np.ones((Npp, Mpp))
        self.I_p_whole = yp.asarray(self.I_p_whole, backend=sol_back_end)
        
        # Compute transfer function
        Pupil_obj = np.zeros((self.Nc,self.Mc))
        frc = (fxx_c**2 + fyy_c**2)**(1/2)
        Pupil_obj[frc<NA_obj/lambda_f] = 1
        T_incoherent = abs(fft2(abs(ifft2(Pupil_obj))**2))
        self.T_incoherent = T_incoherent/np.max(T_incoherent)
        self.T_incoherent = yp.asarray(self.T_incoherent, backend=sol_back_end)
        
        # Compute support function
        self.Pattern_support = np.zeros((Npp,Mpp))
        frp = (fxxp**2 + fyyp**2)**(1/2)
        self.Pattern_support[frp<2*NAs/lambda_f] = 1
        self.Pattern_support =yp.asarray(self.Pattern_support, backend=sol_back_end)

        self.Object_support = np.zeros((self.Nc,self.Mc))
        self.Object_support[frc<2*(NA_obj+NAs)/lambda_f] = 1
        self.Object_support = yp.asarray(self.Object_support, backend=sol_back_end)

        self.OTF_support = np.zeros((self.Nc,self.Mc))
        self.OTF_support[frc<2*NA_obj/lambda_f] = 1
        self.OTF_support = yp.asarray(self.OTF_support, backend=sol_back_end)
        
        # iteration error
        self.err = np.zeros(self.itr+1)

    
    def iterative_algorithm(self, I_image_up, update_shift=1, update_OTF=0):
        if self.sol_back_end == 'numpy':
            F = lambda x: fft2(x)
            iF = lambda x: ifft2(x)
            pad = lambda x, NN, MM: np.pad(x, ((NN,),(MM,)), mode='constant')
        else:
            F = lambda x: af.signal.fft2(x)
            iF = lambda x: af.signal.ifft2(x)
            pad = lambda x, NN, MM: af_pad(x, NN, MM, 0)
        
        f1,ax = plt.subplots(1,3,figsize=(15,5))
        
        tic_time = time.time()
        print('|  Iter  |  error  |  Elapsed time (sec)  |')

        for i in range(0,self.itr):

            # sequential update
            for j in range(0,self.Nimg):

                Ip_shift = yp.real(iF(F(self.I_p_whole) * \
                                      yp.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] +\
                                                                 self.fyyp * self.yshift[j]))))
                Ip_shift[Ip_shift<0]=0
                I_p = Ip_shift[self.yshift_max:self.Nc+self.yshift_max, self.xshift_max:self.Mc+self.xshift_max]
                I_image_current = yp.asarray(I_image_up[j], backend=self.sol_back_end)
                I_multi_f = F(I_p * self.I_obj)
                I_est = iF(self.T_incoherent * I_multi_f)
                I_diff = I_image_current - I_est[self.N_bound_pad:self.N_bound_pad+self.N,\
                                                 self.N_bound_pad:self.N_bound_pad+self.M]

                I_temp = iF(self.T_incoherent * F(pad(I_diff, self.N_bound_pad, self.N_bound_pad)))
                
                # gradient computation
                
                grad_Iobj = -yp.real(I_p * I_temp)
                grad_Ip = -yp.real(iF(F(pad(self.I_obj * I_temp, self.yshift_max, self.xshift_max))\
                                         * yp.exp(-1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] + self.fyyp * self.yshift[j]))))
                if update_OTF ==1:
                    grad_OTF = -np.conj(I_multi_f) * F(I_temp) 

                # updating equation
                self.I_obj = yp.real(iF(F(self.I_obj - grad_Iobj/(yp.max(I_p)**2)) * self.Object_support))
                self.I_p_whole = yp.real(iF(F(self.I_p_whole - grad_Ip/(yp.max(self.I_obj)**2)) * self.Pattern_support))
                
                if update_OTF ==1:
                    self.T_incoherent = self.T_incoherent - grad_OTF/yp.max(yp.abs(I_multi_f)) * \
                         yp.abs(I_multi_f) / (yp.abs(I_multi_f)**2 + 1e-3) / 12 * self.OTF_support

                # shift estimate
                if update_shift ==1:
                    Ip_shift_fx = iF(F(self.I_p_whole) * (1j*2*np.pi*self.fxxp) * \
                                       yp.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] \
                                                                  + self.fyyp * self.yshift[j])))
                    Ip_shift_fy = iF(F(self.I_p_whole) * (1j*2*np.pi*self.fyyp) * \
                                       yp.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[j] \
                                                                  + self.fyyp * self.yshift[j])))
                    Ip_shift_fx = Ip_shift_fx[self.yshift_max:self.yshift_max+self.Nc,\
                                              self.xshift_max:self.xshift_max+self.Mc]
                    Ip_shift_fy = Ip_shift_fy[self.yshift_max:self.yshift_max+self.Nc,\
                                              self.xshift_max:self.xshift_max+self.Mc]

                    grad_xshift = -yp.real(yp.sum(yp.conj(I_temp) * self.I_obj * Ip_shift_fx))
                    grad_yshift = -yp.real(yp.sum(yp.conj(I_temp) * self.I_obj * Ip_shift_fy))

                    self.xshift[j] = self.xshift[j] - \
                    yp.asarray(grad_xshift/self.N/self.M/(yp.max(self.I_obj)**2),backend='numpy')
                    self.yshift[j] = self.yshift[j] - \
                    yp.asarray(grad_yshift/self.N/self.M/(yp.max(self.I_obj)**2),backend='numpy')

                self.err[i+1] += yp.asarray(np.sum(np.abs(I_diff)**2),backend='numpy')

            # Nesterov acceleration
            temp = self.I_obj.copy()
            temp_Ip = self.I_p_whole.copy()
            if i == 0:
                t = 1

                self.I_obj = temp.copy()
                tempp = temp.copy()

                self.I_p_whole = temp_Ip.copy()
                tempp_Ip = temp_Ip.copy()
            else:
                if self.err[i] >= self.err[i-1]:
                    t = 1

                    self.I_obj = temp.copy()
                    tempp = temp.copy()

                    self.I_p_whole = temp_Ip.copy()
                    tempp_Ip = temp_Ip.copy()
                else:
                    tp = t
                    t = (1 + (1 + 4 * tp**2)**(1/2))/2

                    self.I_obj = temp + (tp - 1) * (temp - tempp) / t
                    tempp = temp.copy()

                    self.I_p_whole = temp_Ip + (tp - 1) * (temp_Ip - tempp_Ip) / t
                    tempp_Ip = temp_Ip.copy()

            if np.mod(i,1) == 0:
                print('|  %d  |  %.2e  |   %.2f   |'%(i,self.err[i+1],time.time()-tic_time))
                if i != 1:
                    ax[0].cla()
                    ax[1].cla()
                    ax[2].cla()
                ax[0].imshow(np.maximum(0,np.array(self.I_obj)),cmap='gray');
                ax[1].imshow(np.maximum(0,np.array(self.I_p_whole)),cmap='gray')
                ax[2].plot(self.xshift,self.yshift,'bo')
                display.display(f1)
                display.clear_output(wait=True)
                time.sleep(0.0001)


