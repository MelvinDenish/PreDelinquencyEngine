$ErrorActionPreference = "Stop"

Write-Host "PDI n8n Pipeline Activation" -ForegroundColor Cyan
Write-Host "===========================" -ForegroundColor Cyan

# Login and get cookie
$loginJson = @{ emailOrLdapLoginId = "lmelvindenish@gmail.com"; password = "PdiAdmin@2026!" } | ConvertTo-Json
try {
    $loginResp = Invoke-WebRequest -Uri "http://localhost:5678/rest/login" -Method POST -Body $loginJson -ContentType "application/json" -UseBasicParsing
} catch {
    Write-Host "[ERROR] Cannot reach n8n at localhost:5678. Is pdi-n8n container running?" -ForegroundColor Red
    exit 1
}
$setCookie = $loginResp.Headers["Set-Cookie"]
if ($setCookie -match '^([^=]+)=([^;]+)') {
    $cookieHeader = "$($matches[1])=$($matches[2])"
}
Write-Host "[OK] Logged in as lmelvindenish@gmail.com" -ForegroundColor Green

# Get workflow details (versionId)
$headers = @{ "Cookie" = $cookieHeader }
$wfResp = Invoke-WebRequest -Uri "http://localhost:5678/rest/workflows/Z7mcsQeDA0Bac8hm" -Method GET -UseBasicParsing -Headers $headers
$wfObj = $wfResp.Content | ConvertFrom-Json
$versionId = $wfObj.data.versionId
Write-Host "[OK] Workflow versionId: $versionId" -ForegroundColor Green

# Sync publishedVersionId in SQLite via docker exec
$sqlScript = @"
const sqlite3 = require('/usr/local/lib/node_modules/n8n/node_modules/.pnpm/sqlite3@5.1.7/node_modules/sqlite3');
const db = new sqlite3.Database('/home/node/.n8n/database.sqlite');
const vid = '$versionId';
const wid = 'Z7mcsQeDA0Bac8hm';
db.run('UPDATE workflow_published_version SET publishedVersionId=? WHERE workflowId=?', [vid, wid], function(e) {
  if (e) { console.error(e.message); }
  if (this.changes === 0) {
    db.run('INSERT INTO workflow_published_version (workflowId, publishedVersionId) VALUES (?,?)', [wid, vid]);
  }
  console.log('published version synced:', vid);
  db.close();
});
"@
docker exec pdi-n8n node -e $sqlScript
Write-Host "[OK] Published version synced" -ForegroundColor Green

# Activate the workflow
$actBody = @{ versionId = $versionId } | ConvertTo-Json
$actResp = Invoke-WebRequest -Uri "http://localhost:5678/rest/workflows/Z7mcsQeDA0Bac8hm/activate" -Method POST -Body $actBody -ContentType "application/json" -UseBasicParsing -Headers $headers
$actObj = $actResp.Content | ConvertFrom-Json
if ($actObj.data.active) {
    Write-Host "[OK] Workflow ACTIVATED - pipeline runs every 10 seconds" -ForegroundColor Green
} else {
    Write-Host "[WARN] Unexpected response: $($actResp.Content)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  PDI Live Pipeline is ACTIVE!" -ForegroundColor Green
Write-Host "  n8n UI:      http://localhost:5678" -ForegroundColor White
Write-Host "  Frontend:    http://localhost:3001" -ForegroundColor White
Write-Host "  ML API:      http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "  Start frontend: cd frontend; npm run dev -- --port 3001" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
