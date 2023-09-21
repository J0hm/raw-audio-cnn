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
        samparr = segment.get_array_of_samples()
        samples.append(segment._spawn(samparr[:22050]))
        head = head+step_len

    return samples

def get_batches(samples, batch_size=32):
    batches = []
    for s in samples: # TODO this is so bad need to find a better solution
        # maybe load orignal file in pytorch and split by samples or something?
        s.export("temp.wav", format="wav")
        waveform, sample_rate = torchaudio.load("temp.wav")
        batches += waveform
    
    batches = [torch.stack(batches[n:n+batch_size])[:,None,:] for n in range(0, len(batches)-batch_size, 32)]
    return batches


def get_likely_index(output):
    s, _ = torch.sort(output)
    print(s)
    return output.argmax(dim=-1)


if __name__ == '__main__':
    mmangr = ModelManager()

    audio = pydub.AudioSegment.from_wav(wav_file)
    samples = sample_audio(audio)
    data = get_batches(samples)

    #print(str(mmangr))
    print(mmangr.filter_models("marine"))
    model, loader = mmangr.load_model(20, 8000, "cpu")
    model.eval()

    for batch in data:
        batch = loader.transform(batch)
        output = model(batch)
        print(get_likely_index(output))

    


