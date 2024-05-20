import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

# Location of original source
azimuth = 61 / 180.0 * np.pi  # 61 degrees
elevation = 30 / 180.0 * np.pi  # 30 degrees
distance = 3.0  # 3 meters
dim = 3  # dimensions (2 or 3)
room_dim = np.r_[10.0, 10.0, 10.0]

# algorithms parameters
SNR = 0.0  # signal-to-noise ratio
c = 343.0  # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size

# compute the noise variance
sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2

# Create an anechoic room
aroom = pra.AnechoicRoom(dim, fs=fs, sigma2_awgn=sigma2)

# add the source
source_location = room_dim / 2 + distance * np.r_[
     np.cos(azimuth) * np.cos(elevation),
     np.sin(azimuth) * np.cos(elevation),
     np.sin(elevation)]
print("  Actual location: ", source_location)
source_signal = np.random.randn((nfft // 2 + 1) * nfft)
aroom.add_source(source_location, signal=source_signal)

# We use a circular array with radius 15 cm # and 12 microphones
R_2D = pra.circular_2D_array(room_dim[0:1] / 2, 12, 0.0, 0.15)
# np.full((1, 12), 3))
R_3D = np.vstack((R_2D, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2]))
print("  Microphone positions:", R_3D)
aroom.add_microphone_array(pra.MicrophoneArray(R_3D, fs=aroom.fs))

# run the simulation
aroom.simulate()

# Compute the STFT frames needed
X = np.array(
    [
        pra.transform.stft.analysis(signal, nfft, nfft // 2).T
        for signal in aroom.mic_array.signals
    ]
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


# Construct the new DOA object
doa = pra.doa.SRP(R_3D, fs, nfft, dim=3, c=c)

# this call here perform localization on the frames in X
doa.locate_sources(X)

est_azimuth = doa.azimuth_recon
est_colatitude = doa.colatitude_recon
r = np.sqrt(source_location[0] ** 2 + source_location[1] ** 2 + source_location[2] ** 2)
x, y, z = sph2cart(est_azimuth, np.pi / 2 - est_colatitude, r)
print("  SRP-PHAT:", x, y, z)
# Plot the results
ax.scatter(x, y, z, label='SRP-PHAT')
# Set plot titles and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
mic_x, mic_y, mic_z = R_3D
ax.scatter(mic_x, mic_y, mic_z, color='r', marker='o', s=1, label='Microphones')
source_x, source_y, source_z = source_location
ax.scatter(source_x, source_y, source_z, color='g', marker='o', label='Actual location')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
# Show the plot
plt.show()

# doa.azimuth_recon contains the reconstructed location of the source
print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
print("  Error:", circ_dist(azimuth, doa.azimuth_recon) / np.pi * 180.0, "degrees")
print("  Recovered elevation:", (np.pi / 2 - doa.colatitude_recon) / np.pi * 180.0, "degrees")
print("  Error:", circ_dist(elevation, np.pi / 2 - doa.colatitude_recon) / np.pi * 180.0, "degrees")
