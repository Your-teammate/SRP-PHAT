from func import *

fs = 44100
t = 10
mode = 'c'
source = np.array([5, 7, 6])
source = source[:, None]
center = [0, 0]
mic_count = 8
phi = 0
dist = 0.1
radius = dist * mic_count/(2 * np.pi)
mic_height = 3
if mode == 'l':
    mic_config = np.vstack((pra.linear_2D_array(center, mic_count, phi, dist), np.ones((1, mic_count)) * mic_height))
elif mode == 'c':
    mic_config = np.vstack((pra.circular_2D_array(center, mic_count, phi, radius), np.ones((1, mic_count)) * mic_height))
else:
    raise ValueError('mode must be either l or c')
s = generate_signal(fs, mic_config, t, source)
srp_phat(mic_config, s, fs)
