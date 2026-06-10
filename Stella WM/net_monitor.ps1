# Monitor de red continuo - caza pérdidas intermitentes
# Pinguea el gateway del edificio Y un servidor de internet a la vez,
# agrega por minuto y registra en log con marca de tiempo.

$gateway = "172.20.130.1"   # router del condominio (red local/edificio)
$internet = "8.8.8.8"        # Google (internet, mas alla del edificio)
$logFile  = "d:\Stella WM\net_monitor_log.txt"

$ping = New-Object System.Net.NetworkInformation.Ping
$timeoutMs = 1000

function Test-Target($addr) {
    try {
        $r = $ping.Send($addr, $timeoutMs)
        if ($r.Status -eq 'Success') { return $r.RoundtripTime }
        else { return -1 }
    } catch { return -1 }
}

# Cabecera
$header = "===== Monitor iniciado: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
Add-Content -Path $logFile -Value $header
Add-Content -Path $logFile -Value "Hora     | GATEWAY(edificio) perd% maxms | INTERNET(8.8.8.8) perd% maxms | ESTADO"

while ($true) {
    $minuteStart = Get-Date
    $gwSent=0; $gwLost=0; $gwMax=0
    $inSent=0; $inLost=0; $inMax=0

    # Durante 60 segundos: 1 ping/seg a cada destino
    while (((Get-Date) - $minuteStart).TotalSeconds -lt 60) {
        $gwSent++
        $g = Test-Target $gateway
        if ($g -lt 0) { $gwLost++ } elseif ($g -gt $gwMax) { $gwMax = $g }

        $inSent++
        $i = Test-Target $internet
        if ($i -lt 0) { $inLost++ } elseif ($i -gt $inMax) { $inMax = $i }

        Start-Sleep -Milliseconds 1000
    }

    $gwPct = [math]::Round(100*$gwLost/$gwSent,1)
    $inPct = [math]::Round(100*$inLost/$inSent,1)

    # Marcar estado
    $estado = "ok"
    if ($inPct -ge 2 -or $gwPct -ge 2 -or $inMax -ge 150 -or $gwMax -ge 150) { $estado = "*** PROBLEMA ***" }
    if ($gwPct -ge 2) { $estado += " [fallo dentro del EDIFICIO]" }
    elseif ($inPct -ge 2) { $estado += " [fallo en INTERNET/proveedor]" }

    $ts = $minuteStart.ToString('HH:mm:ss')
    $line = "{0} | GW: {1,4}% max{2,4}ms | NET: {3,4}% max{4,4}ms | {5}" -f $ts, $gwPct, $gwMax, $inPct, $inMax, $estado
    Add-Content -Path $logFile -Value $line
}
