#### https://pyob.oxyry.com/
import numpy as np #line:1

N =8 #line:4
d =0.06 #line:5

def beam_filter (freq_vector ,N ,d ,theta =0 ,mic_id :int =0 ):#line:7

    ""#line:19

    x = (mic_id - (N - 1)/2) *d #line:22


    return np.exp( -1j *2 * np.pi * freq_vector/340 * x *np.cos(theta *np.pi /180 )) #line:24


def beamformer (buffer, thetas, F0, Fs) :

    ## a and b
    BLK , N = np.shape(buffer) 

    F = np.arange(0 , Fs, Fs/BLK)
    FFT = np.fft.fft(buffer ,axis= 0)

    ## c and d 
    k0 = np.abs(F - F0).argmin()

    f = F[k0]
    Mf = FFT[k0, :]

    Y = np.zeros((N, 1),dtype= np.complex_)
    amps = np.zeros((len(thetas), 1), dtype= np.complex_)

    Wn = np.zeros(N, dtype=complex)
    for i ,theta in enumerate(thetas):

        for n in np.arange(0 ,N):

            Wn[n] = beam_filter(f, N, d, theta = theta, mic_id= n)
            
            Y[n, :]= Mf[n] * Wn[n] 

        amps[i, :] = sum(Y, 1)

    P =np.square(np.abs(amps))

    return P


if __name__ =="__main__":
    print ("Simulation")
    beamformer (1)