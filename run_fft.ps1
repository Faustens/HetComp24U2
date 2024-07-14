# PowerShell-Skript zur Ausführung von FFT-Programmen auf verschiedenen WAV-Dateien und Speichern der Ergebnisse in einer CSV-Datei

# Liste der WAV-Dateien
$wavFiles = @("sine_500Hz.wav", "sine_rising_amplitude.wav", "white_noise.wav", "combined_sine_waves.wav")

# CSV-Datei, in die die Ergebnisse geschrieben werden
$outputCsv = "fft_results.csv"

# Kopfzeile für die CSV-Datei
$csvHeader = "File,Method,Output"

# Schreiben der Kopfzeile in die CSV-Datei
$csvHeader | Out-File -FilePath $outputCsv -Encoding utf8

# Schleife über jede WAV-Datei
foreach ($wavFile in $wavFiles) {
    # Ausführen des multi-threaded FFT-Skripts
    $multiThreadResult = py ".\fft multi_thread cpu.py" $wavFile 512 16 50
    $multiThreadOutput = "Multi-Thread,$multiThreadResult"
    
    # Schreiben des Ergebnisses in die CSV-Datei
    "$wavFile,$multiThreadOutput" | Out-File -FilePath $outputCsv -Encoding utf8 -Append
    
    # Ausführen des single-threaded FFT-Skripts
    $singleThreadResult = py ".\fft single_thread.py" $wavFile 512 16 50
    $singleThreadOutput = "Single-Thread,$singleThreadResult"
    
    # Schreiben des Ergebnisses in die CSV-Datei
    "$wavFile,$singleThreadOutput" | Out-File -FilePath $outputCsv -Encoding utf8 -Append
}

Write-Host "FFT-Analyse abgeschlossen. Ergebnisse wurden in $outputCsv gespeichert."
