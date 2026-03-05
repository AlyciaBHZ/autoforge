param(
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

$Package = if ([string]::IsNullOrWhiteSpace($Version)) { "autoforgeai" } else { "autoforgeai==$Version" }
$Indexes = @(
    "https://pypi.org/simple",
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple"
)

Write-Host "Try installing $Package with multiple PyPI mirrors..."

$Success = $false
foreach ($Index in $Indexes) {
    $TrustedHost = ([uri]$Index).Host
    Write-Host "`nTrying: $Index"
    $Args = @(
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "--default-timeout",
        "15",
        "--retries",
        "2",
        "--index-url",
        $Index,
        "--trusted-host",
        $TrustedHost,
        "--progress-bar",
        "off",
        $Package
    )
    $Process = Start-Process -FilePath "python" -ArgumentList $Args -NoNewWindow -Wait -PassThru
    if ($Process.ExitCode -eq 0) {
        $Success = $true
        Write-Host "Installed successfully from $Index"
        break
    }
    Write-Host "Failed on $Index (exit $($Process.ExitCode))"
}

if (-not $Success) {
    Write-Host "`nAll public indexes failed. Falling back to local editable install..."
    if (Test-Path "pyproject.toml") {
        Write-Host "Detected local project source; running: python -m pip install -e ."
        python -m pip install -e .
        Write-Host "Local editable install succeeded."
        exit 0
    }
    Write-Host "Local source not detected. Please check network or run with admin network access."
    exit 1
}
