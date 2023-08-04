import subprocess
import math

processes = []

start = 550
end = 1110
interval = 25
command_len = math.ceil((end - start) / interval)

command_template = 'python3 spectrogram_production.py --csv-file ncs.csv --start-idx %d --end-idx %d'

for i in range(command_len):
    start_idx = i * interval + start
    end_idx = min((i + 1) * interval + start, end)
    processes.append(subprocess.Popen(command_template % (start_idx, end_idx), shell=True))

for process in processes:
    process.wait()
