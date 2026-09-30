[CmdletBinding()]
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VectorStore = Join-Path $ProjectRoot "data\vector_store"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Ambiente Python non trovato: $PythonExe"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env") -PathType Leaf)) {
    throw "File .env non trovato nella radice del progetto."
}
foreach ($name in @("dense.index", "chunks.json")) {
    if (-not (Test-Path -LiteralPath (Join-Path $VectorStore $name) -PathType Leaf)) {
        throw "Indice denso incompleto. Manca $name in $VectorStore."
    }
}
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "La porta $Port e' gia' occupata. Arrestare l'altra configurazione prima di avviare E3."
}
try {
    $ollamaTags = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
}
catch {
    throw "Ollama non risponde. Eseguire prima 00_avvia_servizi.ps1."
}
if ("rag-thesis-ministral-fixed:v4" -notin @($ollamaTags.models | ForEach-Object { $_.name })) {
    throw "Il modello rag-thesis-ministral-fixed:v4 non e' disponibile. Eseguire prima 00_avvia_servizi.ps1."
}

$env:SERVER_HOST = "0.0.0.0"
$env:SERVER_PORT = "$Port"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "rag-thesis-ministral-fixed:v4"
$env:RETRIEVAL_MODE = "hybrid"
$env:RETRIEVAL_TOP_K = "5"
$env:RERANKER_ENABLED = "false"

$addresses = @(Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue |
    Where-Object {
        -not $_.SkipAsSource -and $_.IPAddress -ne "127.0.0.1" -and
        $_.IPAddress -notlike "169.254.*" -and $_.InterfaceAlias -notmatch "Loopback|vEthernet|WSL|Docker"
    } |
    Select-Object -ExpandProperty IPAddress -Unique)
Write-Host "Avvio E3: recupero ibrido BM25 + FAISS con fusione RRF."
Write-Host "Endpoint locale: http://127.0.0.1:$Port/troubleshoot"
foreach ($address in $addresses) { Write-Host "Endpoint di rete: http://${address}:$Port/troubleshoot" }
Write-Host "Premere Ctrl+C per arrestare l'applicazione."

Push-Location $ProjectRoot
try {
    & $PythonExe -m app.main
    if ($LASTEXITCODE -ne 0) {
        throw "L'applicazione e' terminata con codice $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
