import os

from pydub import AudioSegment

sample_rate=44100
input = "recorded"
inpath = "C:/Users/Ethan/Documents/Spectra/"+input
output = "audio"

def to_wav(input_file, output_file, sample_rate=44100):
    """Convert file to wav file

    Args:
        input_file (str): Input audio file
        output_file (str): Output audio file
        sample_rate (int, optional): sample rate of output. Defaults to 44100.
    """
    try:
        format = input_file.split(".")[-1]
        audio = AudioSegment.from_file(input_file, format=format)
        audio = audio.set_frame_rate(sample_rate)
        audio.export(output_file, format="wav")
        print("Conversion completed successfully.")
    except Exception as e:
        print(f"Error converting audio: {e}")

def gather_files_rec(input_path, file_list):
    """Gather files recursively

    Args:
        input_path (str): input path to search
        file_list (list<str>): output file list
    """
    files = [os.path.join(input_path,file) for file in os.listdir(input_path)]
    for file in files:
        if(os.path.isdir(file)):
            gather_files_rec(file, file_list)
        else:
            file_list.append(file)

def find_unconverted(files):
    """Gather all files that have not been converted yet and only convert those

    Args:
        files (list<str>): List of files found
    """
    not_wavs = [x for x in files if x.find(".wav")==-1]
    for file in not_wavs:
        wav = "".join(file.split(".")[:-1])+".wav"
        if files.find(wav)!=-1:
            files.remove(file)

if __name__ == "__main__":
    """Test
    """
    
    files = []
    gather_files_rec(inpath, files) #gather all files
    # find_unconverted(files) #reduce to only files without matching wav

    for file in files:
        wav = ("".join(file.split(".")[:-1])+".wav").replace(input, output)
        if not os.path.exists(os.path.dirname(wav)):
            os.makedirs(os.path.dirname(wav))
        print(f"{file} converted to {wav}")
        to_wav(file, wav, sample_rate)

