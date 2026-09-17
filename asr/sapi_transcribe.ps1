param (
    [Parameter(Mandatory=$true)]
    [string]$WavPath
)

try {
    Add-Type -AssemblyName System.Speech
    $sre = New-Object System.Speech.Recognition.SpeechRecognitionEngine
    $grammar = New-Object System.Speech.Recognition.DictationGrammar
    $sre.LoadGrammar($grammar)
    $sre.SetInputToWaveFile($WavPath)
    $result = $sre.Recognize((New-Object System.TimeSpan(0, 0, 8)))
    if ($result) {
        Write-Output $result.Text
    }
    $sre.Dispose()
} catch {
    Write-Error $_.Exception.Message
}
