$configfiles = "./maze2.yml"
conda activate enbridge
foreach ($conf in $configfiles)
{
	$ScriptBlock = {
		cd C:\Users\Tristan\enbridge\hardware\python
		python circle_track.py $using:conf
	}
	Start-Job $ScriptBlock
}

While (Get-Job -State "Running")
{
  Get-Job
  Start-Sleep 5
}
Get-Job | Receive-Job
Start-Sleep 10