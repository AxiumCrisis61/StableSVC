import os
import wave


def wav2pcm(input_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wav_path_list = [i for i in os.listdir(input_dir) if i[-3:] == 'wav']
    for wav_name in wav_path_list:
        with open(os.path.join(input_dir, wav_name), 'rb') as wavfile:
            ori_data = wavfile.read()
            wavfile.close()
        with open(os.path.join(out_dir, wav_name[:-3], 'pcm'), 'wb') as pcmfile:
            pcmfile.write(ori_data)
            pcmfile.close()


if __name__ == '__main__':
    wav2pcm("/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/Opencpop/segments/wavs",
            "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/Opencpop/segments/pcms")
