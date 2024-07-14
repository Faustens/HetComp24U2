import numpy as np
import scipy.io.wavfile as wavfile
import sys
import multiprocessing
import time

def compute_fft_thread(data,amplitude_sums,block_size,offset):
    local_amplitude_sums = np.zeros(block_size//2)
    #print("Start")
    start_idx = 0
    while start_idx+block_size<len(data):
        end_idx = start_idx+block_size

        block = data[start_idx:end_idx]
        fft_result = np.fft.fft(block)
        amplitudes = np.abs(fft_result[:block_size // 2])
        local_amplitude_sums += amplitudes

        start_idx += offset

    # Amplituden aufsummieren
    with amplitude_sums.get_lock():
        for i in range(len(local_amplitude_sums)):
            amplitude_sums[i] += local_amplitude_sums[i]
    #print("Done")

def compute_fft(file_path, block_size, offset, threshold, thread_num):
    # Lesen der WAV-Datei
    sample_rate, data = wavfile.read(file_path)
    # Sicherstellen, dass es sich um ein eindimensionales Signal handelt (Mono)
    if len(data.shape) > 1:
        data = data[:, 0]
    num_blocks = (len(data) - block_size) // offset
    blocks_per_thread = num_blocks // thread_num

    intervals = []
    start_idx = 0
    for i in range(thread_num):
        end_idx = start_idx + (blocks_per_thread-1)*offset
        intervals.append((start_idx,end_idx))
        start_idx=end_idx

    amplitude_sums = multiprocessing.Array('d', np.zeros(block_size // 2))
    processes = []

    for i in range(thread_num):
        process = multiprocessing.Process(target=compute_fft_thread, args=(data[intervals[i][0]:intervals[i][1]],amplitude_sums,block_size,offset))
        processes.append(process)
        process.start()
        
    for process in processes:
        process.join()

    # Mittelwert der Amplituden berechnen
    amplitude_sums = np.frombuffer(amplitude_sums.get_obj(), dtype=np.float64)
    amplitude_means = amplitude_sums / num_blocks
    """
    # Frequenzanteile ausgeben, deren Amplitudenmittelwert größer als der Schwellwert ist
    frequencies = np.fft.fftfreq(block_size, d=1/sample_rate)[:block_size // 2]
    for freq, amp in zip(frequencies, amplitude_means):
        if amp > threshold:
            print(f"Frequency: {freq:.2f} Hz, Amplitude Mean: {amp:.2f}")
    """

if __name__ == "__main__":
    # Überprüfen, ob die richtigen Anzahl an Argumenten übergeben wurde
    if len(sys.argv) != 5:
        print("Usage: python script.py <path_to_wav> <block_size> <offset> <threshold>")
        sys.exit(1)
    
    # Programmparameter einlesen
    file_path = sys.argv[1]
    block_size = int(sys.argv[2])
    offset = int(sys.argv[3])
    threshold = float(sys.argv[4])
    thread_num = 20
    
    # Gültigkeit der Parameter überprüfen
    if block_size < 64 or block_size > 512:
        print("Block size must be between 64 and 512.")
        sys.exit(1)
    if offset < 1 or offset > block_size:
        print("Offset must be between 1 and block size.")
        sys.exit(1)
    
    # FFT-Berechnung durchführen
    t_total = 0
    for i in range(5):
        t_start = time.time()
        compute_fft(file_path, block_size, offset, threshold, thread_num)
        t_end = time.time()
        t_total += (t_end - t_start)

    t_diff = t_total/5
    print(f"{t_diff}")
