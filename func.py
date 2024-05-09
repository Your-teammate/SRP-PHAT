import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra


def srp_phat(mic_config, s, fs, nFFT=1024):
    hop_size = nFFT // 2
    num_mics = s.shape[1]
    # Calculate the total number of frames needed
    num_frames = np.ceil(s.shape[0] / hop_size).astype(int)
    # Calculate the length of the signal after padding
    padded_length = num_frames * hop_size
    # Initialize a padded array for the signals
    padded_s = np.zeros((padded_length, num_mics))
    # Pad each signal and populate the padded_s array
    for mic in range(num_mics):
        signal = s[:, mic]
        # Ensure the signal is the correct length for the FFT process
        padded_signal = np.pad(signal, (0, padded_length - signal.shape[0]), mode='constant', constant_values=(0, 0))
        padded_s[:, mic] = padded_signal
    # Compute the STFT for each signal
    s_FFT_list = [pra.stft.analysis(padded_s[:, mic], nFFT, hop_size) for mic in range(num_mics)]
    s_FFT = np.stack(s_FFT_list, axis=0)
    # list of frequency bins used to run DoA
    freq_bins = np.arange(30, 330)
    c = 343.0  # speed of sound
    # SRP
    doa = pra.doa.srp.SRP(mic_config, fs, nFFT, c)  # perform SRP approximation
    # Apply SRP-PHAT
    doa.locate_sources(s_FFT, freq_bins=freq_bins)
    # PLOTTING
    doa.polar_plt_dirac()
    plt.title('SRP-PHAT')
    print('SRP-PHAT')
    plt.show()


def generate_signal(fs, mics, t, source):
    # Number of microphones
    a = mics.shape[1]
    # Generate a random signal
    s2 = np.random.rand(fs * t)
    # Calculate distances from the source to each microphone
    dist = np.sqrt(np.sum((mics - source) ** 2, axis=0))
    # Sort distances and calculate time delays
    dist_sorted_indices = np.argsort(dist)
    dist_sorted = dist[dist_sorted_indices]
    # Calculate differential distances between microphones
    dd = np.abs(np.diff(dist_sorted))
    # Speed of sound in m/s
    v_s = 343
    # Calculate time delays
    dt = dd / v_s
    # Convert time delays to sample delays
    xes = np.round(fs * dt).astype(int)
    # Initialize the signal matrix
    signal = np.zeros((a, fs * t))
    # Assign the random signal to the first microphone
    signal[dist_sorted_indices[0], :] = s2
    # Assign the delayed signals to the other microphones
    for i in range(1, a):
        zeros_padding = np.zeros(xes[i-1])
        s_aux = np.concatenate((zeros_padding, s2[:-xes[i-1]]))
        signal[dist_sorted_indices[i], :] = s_aux
    return signal.T
