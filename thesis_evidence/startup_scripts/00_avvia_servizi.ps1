[CmdletBinding()]
param(
    [string]$DemoPath = "C:\Projects\opentelemetry-demo",
    [string]$ModelName = "rag-thesis-ministral-fixed:v4",
    [int]$DockerTimeoutSeconds = 180,
    [int]$DemoTimeoutSeconds = 300,
    [int]$OllamaTimeoutSeconds = 60,
    [switch]$ConfiguraFirewall,
    [switch]$IgnoraVersioneDemo
)

$ErrorActionPreference = "Stop"
$ExpectedDemoCommit = "dedc0178918e260823323b8d95005a8cb924b007"
$ApiPort = 8000
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Modelfile = Join-Path $ProjectRoot "thesis_evidence\configurazione\Modelfile.ministral-v4"

function Resolve-Executable {
    param([string]$Name, [string[]]$Candidates)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    return $null
}

function Test-DockerReady {
    & $script:DockerExe info *> $null
    return $LASTEXITCODE -eq 0
}

function Test-OllamaReady {
    try {
        Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Wait-Until {
    param(
        [scriptblock]$Condition,
        [int]$TimeoutSeconds,
        [string]$FailureMessage
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Condition) {
            return
        }
        Start-Sleep -Seconds 2
    }
    throw $FailureMessage
}

if (-not (Test-Path -LiteralPath $DemoPath -PathType Container)) {
    throw "Repository OpenTelemetry Demo non trovato in '$DemoPath'. Passare il percorso con -DemoPath."
}
if (-not (Test-Path -LiteralPath (Join-Path $DemoPath "compose.yaml") -PathType Leaf)) {
    throw "compose.yaml non trovato in '$DemoPath'."
}
if (-not (Test-Path -LiteralPath $Modelfile -PathType Leaf)) {
    throw "Modelfile non trovato: $Modelfile"
}

$gitExe = Resolve-Executable "git" @()
if ($null -ne $gitExe) {
    $demoCommit = (& $gitExe -C $DemoPath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Impossibile leggere il commit di OpenTelemetry Demo."
    }
    if (-not $IgnoraVersioneDemo -and $demoCommit -ne $ExpectedDemoCommit) {
        throw "OpenTelemetry Demo e' al commit $demoCommit; per la tesi serve $ExpectedDemoCommit. Usare -IgnoraVersioneDemo solo per prove non confrontabili."
    }
    $dirtyFiles = & $gitExe -C $DemoPath status --short
    if ($dirtyFiles) {
        Write-Warning "Il checkout OpenTelemetry Demo contiene modifiche locali."
    }
}

$script:DockerExe = Resolve-Executable "docker" @(
    "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
    "$env:LOCALAPPDATA\Docker\resources\bin\docker.exe"
)
if ($null -eq $script:DockerExe) {
    throw "Docker CLI non trovato. Installare o avviare Docker Desktop e ripetere il comando."
}

if (-not (Test-DockerReady)) {
    $dockerDesktop = Resolve-Executable "Docker Desktop" @(
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
        "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
    )
    if ($null -eq $dockerDesktop) {
        throw "Il motore Docker non risponde e Docker Desktop non e' stato trovato."
    }
    Write-Host "Avvio di Docker Desktop..."
    Start-Process -FilePath $dockerDesktop -WindowStyle Hidden
    Wait-Until -Condition { Test-DockerReady } -TimeoutSeconds $DockerTimeoutSeconds `
        -FailureMessage "Docker non e' diventato disponibile entro $DockerTimeoutSeconds secondi."
}

Write-Host "Avvio dello stack OpenTelemetry Demo..."
Push-Location $DemoPath
try {
    & $script:DockerExe compose up -d --wait --wait-timeout $DemoTimeoutSeconds
    if ($LASTEXITCODE -ne 0) {
        throw "L'avvio di Docker Compose e' terminato con codice $LASTEXITCODE. Controllare docker compose ps."
    }
    & $script:DockerExe compose ps
}
finally {
    Pop-Location
}

$ollamaExe = Resolve-Executable "ollama" @(
    "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
    "$env:LOCALAPPDATA\Ollama\ollama.exe",
    "$env:ProgramFiles\Ollama\ollama.exe"
)
if ($null -eq $ollamaExe) {
    throw "Ollama non trovato. Installarlo e ripetere il comando."
}

if (-not (Test-OllamaReady)) {
    Write-Host "Avvio del server Ollama..."
    $previousOllamaHost = $env:OLLAMA_HOST
    try {
        $env:OLLAMA_HOST = "127.0.0.1:11434"
        Start-Process -FilePath $ollamaExe -ArgumentList @("serve") -WindowStyle Hidden
    }
    finally {
        $env:OLLAMA_HOST = $previousOllamaHost
    }
    Wait-Until -Condition { Test-OllamaReady } -TimeoutSeconds $OllamaTimeoutSeconds `
        -FailureMessage "Ollama non e' diventato disponibile entro $OllamaTimeoutSeconds secondi."
}

$tags = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
$availableModels = @($tags.models | ForEach-Object { $_.name })
if ($ModelName -notin $availableModels) {
    Write-Host "Creazione del modello fisso $ModelName..."
    & $ollamaExe create $ModelName -f $Modelfile
    if ($LASTEXITCODE -ne 0) {
        throw "Creazione del modello Ollama non riuscita."
    }
    $tags = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
    if ($ModelName -notin @($tags.models | ForEach-Object { $_.name })) {
        throw "Ollama non espone il modello $ModelName dopo la creazione."
    }
}

if ($ConfiguraFirewall) {
    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent()
    )
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Per configurare il firewall eseguire PowerShell come amministratore."
    }
    $ruleName = "Tesi RAG API TCP 8000"
    if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
            -Protocol TCP -LocalPort $ApiPort -Profile Private -RemoteAddress LocalSubnet | Out-Null
        Write-Host "Regola firewall creata per TCP $ApiPort sulla rete privata locale."
    }
    else {
        Write-Host "La regola firewall per TCP $ApiPort e' gia' presente."
    }
}

Write-Host ""
Write-Host "Servizi pronti:"
Write-Host "  OpenTelemetry Demo: avviata tramite Docker Compose"
Write-Host "  Ollama: http://127.0.0.1:11434"
Write-Host "  Modello: $ModelName"
Write-Host "Avviare ora uno solo degli script E0-E3 in un'altra finestra PowerShell."
