import torchaudio
import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
from metrics import cer

quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')
file_name = '0'
wav, sr = torchaudio.load('./' + file_name + '.wav')

# Cut into 2-second fragments
for i in range(0, wav.shape[1], 32000):
    wav_temp = wav[:, i:i + 32000]
    torchaudio.save(file_name + '.' + str(i // 32000)+'.wav', wav_temp, sample_rate=sr)

# Get transcription
files = ['./' + file_name + '.'+str(i)+'.cut.wav' for i in range(65)]
total_transcription = ''
synchr = []
portion = []
i = 0
for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
    transcription = transcription.replace(" ", "").replace("'", "")
    for j in range(len(transcription)):
        synchr.append(i)
        portion.append(float(j)/len(transcription))
    i += 1
    total_transcription += transcription

# Parse real transcription
songs = {}
start = {}
finish = {}
line_id = {}
real_transcription = ''
lines = []

def one_song(path, song_id):
    is_open = 0
    i = 0
    real_transcription = ''
    lines = []
    with open(path, 'r') as cur_file:
        cur_line = ''
        cur_start = ''
        cur_finish = ''
        music = '♪'
        for line in cur_file.readlines():
            if line[:3] == '00:':
                cur_start = line.split()[0]
                cur_finish = line.split()[2]
            cnt = line.count(music)
            if cnt == 1:
                if is_open == 0:
                    cur_line += line[1:-1]
                else:
                    cur_line += " " + line[:-2]
                    cur_line = cur_line.strip()
                    cur_line = cur_line.replace(" ", "").replace("'", "").lower()
                    songs[cur_line] = song_id
                    start[cur_line] = cur_start
                    finish[cur_line] = cur_finish
                    line_id[cur_line] = str(i)
                    lines.append(cur_line)
                    real_transcription += cur_line
                    i += 1
                    cur_line = ''
                is_open = (is_open + 1) % 2
            elif cnt == 2:
                cur_line += line[1:-2]
                if len(cur_line) < 2:
                    continue
                cur_line = cur_line.strip()
                cur_line = cur_line.replace(" ", "").replace("'", "").lower()
                songs[cur_line] = song_id
                start[cur_line] = cur_start
                finish[cur_line] = cur_finish
                line_id[cur_line] = str(i)
                lines.append(cur_line)
                real_transcription += cur_line
                i += 1
                cur_line = ''
            elif cnt == 0 and is_open == 1:
                cur_line += " " + line[:-1]
    return real_transcription, lines

real_transcription, lines = one_song(file_name + '.en.vtt', file_name)


# Beam search
val = 0
coef = len(total_transcription)/len(real_transcription)
tengram = [total_transcription[i:i+10] for i in range(len(total_transcription)-10)]
tengram_real = [real_transcription[i:i+10] for i in range(len(real_transcription)-10)]
beam_size = 10000
effective_beam_size = 10000
best_n = [[-1]]*beam_size
error_n = [0]*beam_size
for i in range(0, len(tengram)-5, 5):
    best = 100
    err_j = []
    for j in range(max(0, int(i/coef) - 100), min(int(i/coef)+100, len(tengram_real)), 2):
        err = cer(tengram[i], tengram_real[j])
        err_j.append((err, j))
    err_j.sort()
    if i == 0:
        new_best_n = [[-1]*1]*beam_size
        effective_beam_size = min(len(err_j), beam_size)
        for k1 in range(effective_beam_size):
            new_best_n[k1] = [err_j[k1][1]]
            error_n[k1] += err_j[k1][0]
        best_n = new_best_n
    else:
        to_choose = []
        for k1 in range(effective_beam_size):
            for k2 in range(min(len(err_j), beam_size)):
                if best_n[k1][-1] < err_j[k2][1]:
                    to_choose.append((error_n[k1]+err_j[k2][0], k1, err_j[k2][1]))
        if len(to_choose) == 0:
            break
        to_choose.sort()
        effective_beam_size = min(beam_size, len(to_choose))
        new_best_n = [[0]*(i+2)]*effective_beam_size
        k1 = 0
        k2 = 0
        while k1 < effective_beam_size:
            if k1 > 0 and to_choose[k1][0] == to_choose[k1-1][0] and to_choose[k1][2] == to_choose[k1-1][2]:
                k1 += 1
            else:
                error_n[k2] = to_choose[k1][0]
                new_best_n[k2] = best_n[to_choose[k1][1]] + [to_choose[k1][2]]
                k2 += 1
                k1 += 1
        effective_beam_size = k2
        best_n = new_best_n

# Cut audio by line
import torch
starts = [0]
for line in lines:
    line = line.replace(" ", "").strip()
    starts.append(starts[-1]+len(line))
gen_starts = []
i = 0
for cur_start in starts[:-1]:
    while i < len(best_n[0]) and best_n[0][i] < cur_start:
        i += 1
    if i == len(best_n[0]):
        i -= 1
    if best_n[0][i] == cur_start:
        gen_starts.append(i*5)
    else:
        gen_starts.append(int(i*5 - 5*(cur_start - best_n[0][i])/(best_n[0][i] - best_n[0][i-1])))
start_fragment = 0
end_fragment = 0
for i in range(len(gen_starts)):
    start_fragment = max(0, synchr[gen_starts[i]]*32000 + int(portion[gen_starts[i]]*32000)-8000)
    end_fragment = synchr[gen_starts[i]+int(0.7*len(lines[i]))]*32000 + int(portion[gen_starts[i]+int(0.7*len(lines[i]))]*32000)+8000
    start_fragment = max(0, get_secs(start[lines[i]])-8000)
    end_fragment = get_secs(finish[lines[i]])+32000
    total_wav = wav[:, start_fragment:end_fragment]
    torchaudio.save(file_name + '.'+str(i)+'.cut.wav', total_wav, sr)
print('Success')