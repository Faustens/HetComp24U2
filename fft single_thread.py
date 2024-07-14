import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft
import sys
import time

def main():
    if len(sys.argv) != 5:
        print("Usage: python fft_analysis.py <path_to_wav_file> <block_size> <block_offset> <threshold>")
        return

    # Parse command line arguments
    wav_file = sys.argv[1]
    block_size = int(sys.argv[2])
    block_offset = int(sys.argv[3])
    threshold = float(sys.argv[4])

    # Read WAV file
    sample_rate, data = wav.read(wav_file)
    
    # Ensure mono audio (if stereo, take the first channel)
    if len(data.shape) > 1:
        data = data[:, 0]

    # Initialize variables
    num_blocks = (len(data) - block_size) // block_offset + 1
    amplitudes = np.zeros(block_size // 2)  # We only care about the positive frequencies

    # Process each block
    for i in range(num_blocks):
        start = i * block_offset
        end = start + block_size

        # Apply FFT to the current block
        block = data[start:end]
        fft_result = fft(block)
        
        # Compute amplitude spectrum
        amplitude_spectrum = np.abs(fft_result[:block_size // 2])
        
        # Accumulate amplitude values
        amplitudes += amplitude_spectrum

    # Calculate average amplitude
    amplitudes /= num_blocks

    # Print frequencies with amplitudes above the threshold
    """
    frequencies = np.fft.fftfreq(block_size, d=1/sample_rate)[:block_size // 2]
    for freq, amp in zip(frequencies, amplitudes):
        if amp > threshold:
            print(f"Frequency: {freq} Hz, Amplitude: {amp}")
    """
if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    t_diff = (t_end - t_start)
    print(f"{t_diff}")