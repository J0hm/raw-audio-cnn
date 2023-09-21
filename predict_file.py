import pydub
import torchaudio
import torch
from modules.data_management import ModelManager


sample_len = 1 # seconds per sample
sample_rate = 8000 # resample to this sample rate
step_len = sample_len/10 # how far to step through the file each sample

wav_file = "marineaudio/rawaudio/Balaena_mysticetus/8800600C.wav"

def sample_audio(audio):
    duration = audio.duration_seconds

    head = 0
    samples = []
    while(head+sample_len<duration):
        segment = audio[head*1000:(head+sample_len)*1000]
        samples.append(segment)
        head = head+step_len
    return samples

def get_input(samples, batch_size=32):
    sample_arrays = [s.get_array_of_samples()[0:8000] for s in samples]
    data = [torch.Tensor(sample_arrays[n:n+batch_size])[:, None, :] for n in range(0, len(sample_arrays)-batch_size, batch_size)]

    return data





if __name__ == '__main__':
    mmangr = ModelManager()

    audio = pydub.AudioSegment.from_wav(wav_file)
    
    samples = sample_audio(audio)
    data = get_input(samples)

    
    #print(str(mmangr))
    print(mmangr.filter_models("marine"))
    model, loader = mmangr.load_model(20, 8000, "cpu")
    model.eval()
    
    for batch in data:
        #batch = loader.transform(data)
        print(batch.size())
        output = model(data)


