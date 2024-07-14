import numpy as np
import scipy.io.wavfile as wavfile

def generate_sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_white_noise(duration, sample_rate=44100, amplitude=1.0):
    return amplitude * np.random.normal(0, 1, int(sample_rate * duration))

def generate_wav_file(filename, data, sample_rate=44100):
    wavfile.write(filename, sample_rate, data)

def generate_test_files():
    sample_rate = 44100
    duration = 10000
    # Szenario 1: Sinuswellen mit verschiedenen Frequenzen
    for freq in [500]:
        data = generate_sine_wave(freq, duration, sample_rate=sample_rate)
        generate_wav_file(f'sine_{freq}Hz.wav', data, sample_rate)
    
    # Szenario 2: Weißrauschen
    data = generate_white_noise(duration, sample_rate=sample_rate)
    generate_wav_file('white_noise.wav', data, sample_rate)
    
    # Szenario 3: Kombination von Sinuswellen
    combined_data = (generate_sine_wave(440, duration, sample_rate=sample_rate) +
                     generate_sine_wave(880, duration, sample_rate=sample_rate) +
                     generate_sine_wave(1760, duration, sample_rate=sample_rate))
    combined_data /= 3  # Normalisieren, um Verzerrungen zu vermeiden
    generate_wav_file('combined_sine_waves.wav', combined_data, sample_rate)
    
    # Szenario 4: Sinuswelle mit plötzlich ansteigender Amplitude
    data = generate_sine_wave(1000, duration, sample_rate=sample_rate)
    data[int(sample_rate * 2.5):] *= 2  # Amplitude nach der Hälfte der Dauer verdoppeln
    generate_wav_file('sine_rising_amplitude.wav', data, sample_rate)

if __name__ == "__main__":
    generate_test_files()
