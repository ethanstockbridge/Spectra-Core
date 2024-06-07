
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from config.config_manager import ConfigManager


class spectrogram():
    """Holds spectrograph data of audio
    """
    def __init__(self, audio, fs):
        """Initialize the spectrogram

        Args:
            audio (np.array): Audio data
            fs (int): Sample freq
        """
        self.__config = ConfigManager()
        nperseg = self.__config["spectrogram"]["nperseg"]
        noverlap = self.__config["spectrogram"]["noverlap"]
        self.f, self.t, self.Sxx = signal.spectrogram(audio, fs, nperseg=nperseg, noverlap=noverlap)

    def get_values(self):
        """Return spectrogram values

        Returns:
            tuple: f, t, Sxx
        """
        return self.f, self.t, self.Sxx
    def limit_frequencies(self, fmin, fmax):
        """Limit the frequencies in the spectrogram

        Args:
            fmin (int): Min freq
            fmax (int): Max freq

        Returns:
            tuple: f, Sxx
        """
        freq_slice = np.where((self.f >= fmin) & (self.f <= fmax))
        # keep only frequencies of interest
        f = self.f[freq_slice]
        Sxx = self.Sxx[freq_slice,:][0]
        return f, Sxx
    def show(self):
        """Debug function to show the spectrogram

        Args:
            f (array): frequency
            t (array): time
            Sxx (array): Sxx
        """
        plt.figure(figsize=(8,10))
        plt.pcolormesh( 10*np.log(Sxx), cmap='gray')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()