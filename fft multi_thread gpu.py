import numpy as np
import scipy.io.wavfile as wavefile
import pyopencl as cl
import sys

# Dieses Programm funktioniert nicht
# OpenCL-Kernel zum Berechnen der FFT
'''
Berechnet die fft mit einer festgelegten Menge an Kernen. Kerne, die mit einer FFT-Berechnung fertig sind, 
nehmen sich automatisch den nächsten Block, bis num_blocks viele Blöcke berechnet wurden
Input: 
    input: Pointer zum data array
    output_amplitude_sums: pointer zum amplitude_sums array
    task_counter: nummer des als nächstes zu berechnenden Blocks

'''
kernel_code = """
//#define M_PI 3.14159265358979323846

// Hilfsfunktion: Komplexe Multiplikation
inline float2 complex_mult(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Hilfsfunktion: atomic_add für float Werte (leider ungenauer)
inline float atomic_add_float(__global float *addr, float val) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal, oldVal;

    do {
        oldVal.floatVal = *addr;
        newVal.floatVal = oldVal.floatVal + val;
    } while (atomic_cmpxchg((volatile __global int *)addr, oldVal.intVal, newVal.intVal) != oldVal.intVal);

    return oldVal.floatVal;
}

// OpenCL Kernel für FFT eines Blocks und Speicherung der positiven Amplituden
__kernel void fft(__global float *input,
                    __global float *output_amplitude_sums,
                    __global volatile int *task_counter,
                    const int num_blocks,
                    const int block_size,
                    const int offset)
{
    int task_index;
    __local float2 output_fft[512];
    while(true) {
        int task_index = atomic_add(task_counter,1);
        if(task_index >= num_blocks) {
            break;
        }

        // Berechnung des Startindex für den aktuellen Block
        int start = task_index * offset;
        int end = start + block_size;

        // Signal-Block aus dem Eingabearray lesen
        __global float *block = input + start;

        // Array für FFT-Ergebnis
        //__global float2 *output_fft = (__global float2 *) block;

        // Berechnung der FFT mit Cooley-Tukey Algorithmus
        for (int k = 0; k < block_size; k++) {
            float2 sum = 0.0f;
            for (int n = 0; n < block_size; n++) {
                float theta = -2.0f * M_PI * k * n / block_size;
                float2 W = (float2)(cos(theta), sin(theta));
                sum += complex_mult(block[n], W);
            }
            output_fft[k] = sum;
        }

        // Addieren der Ergebnisse zu den bisherigen Amplituden Summen (nur positive Frequenzen)
        for (int k = 0; k < block_size / 2; k++) {
            float amplitude_val = sqrt(output_fft[k].x * output_fft[k].x + output_fft[k].y * output_fft[k].y);
            atomic_add_float(&output_amplitude_sums[k],amplitude_val);;
            //atomic_add((volatile __global float *)&output_amplitude_sums[k],amplitude_val);
        }
    }
}
"""

def main():
    if len(sys.argv) != 5:
        print("Verwendung: python 'fft multi_thread gpu.py' <pfad_zur_wav_datei> <blockgröße> <blockversatz> <schwellenwert>")
        return

    # Kommandozeilenargumente einlesen
    wav_datei = sys.argv[1]
    block_size = int(sys.argv[2])
    offset = int(sys.argv[3])
    threshold = float(sys.argv[4])
    num_kernels = 1024

    # Blockgröße und Versatz validieren
    if block_size < 64 or block_size > 512:
        print("Die Blockgröße muss zwischen 64 und 512 liegen.")
        return
    if offset < 1 or offset > block_size:
        print("Der Blockversatz muss zwischen 1 und der Blockgröße liegen.")
        return

    # WAV-Datei lesen
    sample_rate, data = wavefile.read(wav_datei)
    
    # Sicherstellen, dass es sich um Mono-Audio handelt (wenn Stereo, nur den ersten Kanal verwenden)
    if len(data.shape) > 1:
        data = data[:, 0]

    # Variablen initialisieren
    num_blocks = (len(data) - block_size) // offset + 1
    amplitude_results = np.zeros(block_size//2, dtype=np.float32)  # Wir interessieren uns nur für die positiven Frequenzen
    #task_counter = 0
    task_counter = np.zeros(1, dtype=np.int32)
    print(num_blocks)

    # OpenCL-Kontext und Warteschlange erstellen
    gpu = cl.get_platforms()[0].get_devices()[0]
    #context = cl.create_some_context()
    context = cl.Context(devices=[gpu])
    queue = cl.CommandQueue(context)


    # Buffer für Eingabe- und Ausgabedaten erstellen
    mf = cl.mem_flags
    input_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    output_buffer = cl.Buffer(context, mf.WRITE_ONLY, amplitude_results.nbytes)
    task_counter_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=task_counter)
    #output_buffer = cl.Buffer(context, mf.WRITE_ONLY, num_blocks * block_size * np.float32().itemsize)

    # Programm und Kernel erstellen
    program = cl.Program(context, kernel_code).build()
    fft_kernel = program.fft

    # Kernel-Argumente setzen
    fft_kernel.set_arg(0, input_buffer)
    fft_kernel.set_arg(1, output_buffer)
    fft_kernel.set_arg(2, task_counter_buffer)
    fft_kernel.set_arg(3, np.int32(num_blocks))
    fft_kernel.set_arg(4, np.int32(block_size))
    fft_kernel.set_arg(5, np.int32(offset))

    # Kernel ausführen
    cl.enqueue_nd_range_kernel(queue, fft_kernel, (num_kernels,), None)
    queue.finish()

    # Ergebnis zurücklesen
    result = np.zeros((block_size//2), dtype=np.float32)
    cl.enqueue_copy(queue, result, output_buffer).wait()
    print(len(result))
    print(result)
    # Durchschnittliche Amplitude berechnen
    '''
    amplitude_sums = np.zeros(block_size//2)
    for i in range(num_blocks):
        start = i * offset
        end = start + block_size // 2
        amplitude_sums += result[start:end]
    '''
    amplitude_means = result / num_blocks
    print(amplitude_means)

    # Frequenzen mit Amplituden über dem Schwellenwert ausgeben
    frequenzen = np.fft.fftfreq(block_size, d=1/sample_rate)[:block_size // 2]
    for freq, amp in zip(frequenzen, amplitude_means):
        if amp > threshold:
            print(f"Frequenz: {freq:.2f} Hz, Mittlere Amplitude: {amp:.2f}")

if __name__ == "__main__":
    main()
