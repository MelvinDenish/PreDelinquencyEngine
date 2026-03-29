$ErrorActionPreference = "Stop"

# Login and manually capture cookie
$loginJson = @{ emailOrLdapLoginId = "lmelvindenish@gmail.com"; password = "L.Melvin@07" } | ConvertTo-Json

$resp = Invoke-WebRequest -Uri "http://localhost:5678/rest/login" -Method POST -Body $loginJson -ContentType "application/json" -UseBasicParsing
Write-Host "[OK] Login: HTTP $($resp.StatusCode)"

# Extract auth cookie from Set-Cookie header
$setCookieRaw = $resp.Headers["Set-Cookie"]
Write-Host "Raw Set-Cookie: $($setCookieRaw.Substring(0, [Math]::Min(80, $setCookieRaw.Length)))..."

# Parse cookie name=value
if ($setCookieRaw -match '^([^=]+)=([^;]+)') {
    $cookieName = $matches[1]
    $cookieValue = $matches[2]
    Write-Host "Cookie name: $cookieName"
} else {
    Write-Host "[ERROR] Cannot parse cookie"
    exit 1
}

$cookieHeader = "$cookieName=$cookieValue"

# Read workflow
$wfPath = Join-Path $PSScriptRoot "pdi_live_simulation_workflow.json"
$wfText = [System.IO.File]::ReadAllText($wfPath, [System.Text.Encoding]::UTF8)
$wfObj = $wfText | ConvertFrom-Json
foreach ($prop in @('id', 'active')) {
    if ($wfObj.PSObject.Properties[$prop]) { $wfObj.PSObject.Properties.Remove($prop) }
}
$cleanJson = $wfObj | ConvertTo-Json -Depth 50
$bytes = [System.Text.Encoding]::UTF8.GetBytes($cleanJson)
Write-Host "[OK] Workflow prepared ($($bytes.Length) bytes)"

# Import workflow with manual cookie
$headers = @{ "Cookie" = $cookieHeader }
try {
    $importResult = Invoke-WebRequest -Uri "http://localhost:5678/rest/workflows" -Method POST -Body $bytes -ContentType "application/json; charset=utf-8" -UseBasicParsing -Headers $headers
    Write-Host "[OK] Import: HTTP $($importResult.StatusCode)"
    
    if ($importResult.Content -match '"id"\s*:\s*"([^"]+)"') {
        $wfId = $matches[1]
        Write-Host "[OK] Workflow ID: $wfId"
    } else {
        Write-Host "[ERROR] No workflow ID in response"
        Write-Host $importResult.Content.Substring(0, 200)
        exit 1
    }
} catch {
    Write-Host "[ERROR] Import: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $sr = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        Write-Host "Body: $($sr.ReadToEnd())"
    }
    exit 1
}

# Activate workflow
try {
    $null = Invoke-WebRequest -Uri "http://localhost:5678/rest/workflows/$wfId" -Method PATCH -Body '{"active":true}' -ContentType "application/json" -UseBasicParsing -Headers $headers
    Write-Host "[OK] Workflow ACTIVATED!"
} catch {
    Write-Host "[ERROR] Activate: $($_.Exception.Message)"
    exit 1
}

# Test webhook
Start-Sleep -Seconds 2
try {
    $testResult = Invoke-WebRequest -Uri "http://localhost:5678/webhook/pdi-ingest" -Method POST -Body '{"source":"import_test"}' -ContentType "application/json" -UseBasicParsing
    Write-Host "[OK] Webhook test: HTTP $($testResult.StatusCode)"
} catch {
    Write-Host "[WARN] Webhook test: $($_.Exception.Message)"
}

Write-Host ""
Write-Host "=========================================="
Write-Host "  n8n PDI Pipeline is LIVE!"
Write-Host "  Webhook: http://localhost:5678/webhook/pdi-ingest"
Write-Host "  UI: http://localhost:5678"
Write-Host "=========================================="
