import os
import torchaudio
import nemo
import nemo.collections.asr as nemo_asr
from metrics import cer

quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained('QuartzNet15x5Base-En')

def new_cut(line_, file_):
    best_res = 100
    best_j = -1
    transcription = ''
    files = [file_]
    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        transcription = transcription.replace(" ", "").replace("'", "")
    for j in range(len(transcription)-len(line_)):
        if cer(line_, transcription[j:j+len(line_)]) < best_res:
            best_res = cer(line_, transcription[j:j+len(line_)])
            best_j = j
    if best_j >= 0:
        wav, sr = torchaudio.load(file_)
        length = wav.shape[1]
        left_cut = float(best_j)/len(transcription)
        right_cut = float(best_j+len(line_))/len(transcription)
        wav = wav[:, int(length*left_cut):int(length*right_cut)]
        torchaudio.save(file_, wav, sr)

def get_secs(s):
	n = 0
	s = s.split(':')
	n += int(s[1])*60*16000
	s = s[2].split('.')
	n += int(s[0])*16000
	n += int(s[1])*16
	return n


dataset_folder = '.'

with open(dataset_folder + '/data_final.tsv', 'w+') as fout:
	with open(dataset_folder + '/data.tsv', 'r') as fin:
		lines = fin.readlines()
		for line in lines:
			line = line.split('\t')
			time = line[0].split()
			start_time = max(0, get_secs(time[0]) - 8000)
			finish_time = get_secs(time[2]) + 32000
			file = line[2]
			if file.split('.')[0].split('/')[1] + ".wav" in os.listdir(dataset_folder + "/" + file[0]):
				if file.split('.')[0].split('/')[1] + "." + str(start_time) + ".ogg" in os.listdir(dataset_folder + "/" + file[0]):
					continue
				wav, sr = torchaudio.load(dataset_folder + "/" + file.split('.')[0] + ".wav")
				wav = wav[:, start_time:finish_time]
				torchaudio.save(dataset_folder + "/" + file.split('.')[0] + str(start_time) + ".wav", wav, sr)
				new_cut(line[1].replace(" ", "").replace("'", "").strip(), dataset_folder + "/" + file.split('.')[0] + str(start_time) + ".wav")
                name = file[0] + "/" + file.split('.')[0] + "." + str(start_time) + ".ogg"
				os.system("ffmpeg -i " + dataset_folder + "/" + file.split('.')[0] + str(start_time) + ".wav" + " -acodec libvorbis " + dataset_folder + "/" + file.split('.')[0] + "." + str(start_time) + ".ogg")
				
				line.append(name)
				line = '\t'.join(line)
				fout.write(line)
				fout.write('\n')