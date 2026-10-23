@echo off
title Keep Awake - close this window to stop
set "HOURS="
set /p "HOURS=How many hours to keep the screen on? (e.g. 2 or 1.5, blank = forever): "

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$h = 0.0;" ^
  "if ([string]::IsNullOrWhiteSpace($env:HOURS)) { $end = [datetime]::MaxValue; Write-Host 'Keeping screen on indefinitely. Close this window to stop.' }" ^
  "elseif (-not [double]::TryParse($env:HOURS, [ref]$h) -or $h -le 0) { Write-Host 'Invalid number of hours.'; exit 1 }" ^
  "else { $end = (Get-Date).AddHours($h); Write-Host ('Keeping screen on until ' + $end.ToString('h:mm tt') + '. Close this window to stop early.') };" ^
  "Add-Type -Namespace W -Name M -MemberDefinition '[DllImport(\"user32.dll\")] public static extern void mouse_event(int f, int x, int y, int d, int e);';" ^
  "while ((Get-Date) -lt $end) {" ^
  "  [W.M]::mouse_event(1, 1, 0, 0, 0);" ^
  "  Start-Sleep -Milliseconds 100;" ^
  "  [W.M]::mouse_event(1, -1, 0, 0, 0);" ^
  "  Start-Sleep -Seconds 60" ^
  "};" ^
  "Write-Host 'Done.'"

pause