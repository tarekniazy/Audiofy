import numpy as np

def iirfilter(N, Wn):
    
    Wn = np.asarray(Wn)
    if np.any(Wn <= 0) or np.any(Wn >= 1):
        raise ValueError("Digital filter critical frequencies "
                            "must be 0 < Wn < 1")

    ######## Initalize z,p,k
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1

    ###############################
    fs = 2.0
    warped = 2 * fs * np.tan(np.pi * Wn / fs)  #ANALOG FREQUENCY

    bw = float(warped[1] - warped[0])
    wo = float(np.sqrt(warped[0] * warped[1]))

    ############################# Convert lowpass to bandpass
    degree =  len(p) - len(z)

    # Scale poles and zeros to desired bandwidth
    z1 = z * bw/2 + np.sqrt((z * bw/2)**2 - wo**2)
    z2 = z * bw/2 - np.sqrt((z * bw/2)**2 - wo**2)

    p1 = p * bw/2 + np.sqrt((p * bw/2)**2 - wo**2)
    p2 = p * bw/2 - np.sqrt((p * bw/2)**2 - wo**2)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = np.concatenate((z1, z2))
    p_bp = np.concatenate((p1, p2))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = np.append(z_bp, np.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree
  
    ################################## Bi-Linear Transformation
    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z_bp) / (fs2 - z_bp)
    p_z = (fs2 + p_bp) / (fs2 - p_bp)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    degree = len(p_bp) - len(z_bp)
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k_bp * np.real(np.prod(fs2 - z_bp) / np.prod(fs2 - p_bp))

    ################################## Extract transfer function coefficients
    return coefficients(z_z, p_z, k_z)



def coefficients(z, p, k):
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    if len(z.shape) > 1:
        temp = np.poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * np.poly(z[i])
    else:
        b = k * np.poly(z)
    a = np.atleast_1d(np.poly(p))

    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                         np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                         np.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a