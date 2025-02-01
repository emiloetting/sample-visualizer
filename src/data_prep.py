import os
import shutil
import pandas as pd
import numpy as np
import librosa as lb
import soundfile as sf
import timeit



# Constants
NORMALIZING_LEVEL = -20  # Example value, set this to your actual normalizing level
SILENCE_THRESHOLD_DB = 30  # Example value, set this to your actual silence threshold
FIXED_LENGTH = 2048  # Set this to the desired fixed length for your audio samples
SAMPLE_ROOT_FOLDER = r'C:\Users\emilo\Documents\Splice - bruder_emil\Samples\packs' # Set this to the root folder of your audio samples
OUTPUT_CSV_NAME = 'scaled_data.csv'
CSV_OUTPUT_FOLDER = os.path.join(os.getcwd(), 'audio_data', 'csv_to_cluster')

# Verzeichnisse erstellen
cwd = os.getcwd()  
audio_data_dir = os.path.join(cwd, 'audio_data')
audio_data_all_dir = os.path.join(audio_data_dir, 'all')
os.makedirs(audio_data_all_dir, exist_ok=True)
os.makedirs(os.path.join(audio_data_dir, 'loops'), exist_ok=True)
os.makedirs(os.path.join(audio_data_dir, 'to_label'), exist_ok=True)
os.makedirs(CSV_OUTPUT_FOLDER, exist_ok=True)

print('audio_data_dir:', audio_data_dir)

def aiff_to_wav(file_path):
    aiff_file, sample_rate = sf.read(file_path)
    wav_path = file_path.replace(".aif", ".wav")
    sf.write(wav_path, aiff_file, sample_rate)
    return wav_path

def file_finder_and_sorter(root_folder):
    all_samples = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith("aif"):      #convert aif files to wav
                file = aiff_to_wav(os.path.join(root, file))
            full_path = os.path.join(root, file)
            all_samples.append(full_path)
            shutil.copy(full_path, audio_data_all_dir)  # Copy to 'all' subfolder
            if "loop" in str(file).lower():
                shutil.copy(full_path, os.path.join(audio_data_dir, 'loops'))
            else:
                shutil.copy(full_path, os.path.join(audio_data_dir, 'to_label'))
    return all_samples

def load_audio(file_path):
    audio, sr = lb.load(file_path, sr=None, mono=True)
    return audio, sr

def normalize_amplitude(input_as_array, target_db=NORMALIZING_LEVEL):
    max_amplitude = np.max(np.abs(input_as_array))
    target_amplitude = 10 ** (target_db / 20.0)
    normalized_file = input_as_array * (target_amplitude / max_amplitude)
    return normalized_file

def trim_start_end_silence(input_as_array):
    trimmed_audio, _ = lb.effects.trim(input_as_array, top_db=SILENCE_THRESHOLD_DB)
    return trimmed_audio

def preprocess_sample(input_as_array):
    normalized = normalize_amplitude(input_as_array)
    preprocessed_array = trim_start_end_silence(normalized)
    return preprocessed_array

def truncate_if_needed(input_as_array, length=FIXED_LENGTH):
    if len(input_as_array) > length:
        truncated_array = input_as_array[:length]
        return truncated_array
    return input_as_array

def adjust_n_fft(audio):
    return min(FIXED_LENGTH, len(audio))

def make_mel_db(input_as_array, sample_rate):
    n_fft = min(2048, len(input_as_array))  # Adjust n_fft to be smaller or equal to the input signal length
    print(f"Using n_fft={n_fft} for input signal of length={len(input_as_array)}")  # Debugging information
    mel_spectrogram = lb.feature.melspectrogram(y=input_as_array, sr=sample_rate, n_fft=n_fft)
    mel_db = lb.power_to_db(mel_spectrogram, ref=np.max)
    return mel_db

def preprocess_sample_to_mel_db(input_as_array, sample_rate):
    preprocessed_sample = preprocess_sample(input_as_array)
    truncated_sample = truncate_if_needed(preprocessed_sample)  # Ensure fixed length if needed
    mel_db = make_mel_db(truncated_sample, sample_rate)
    return mel_db

def get_features(input_as_array, sample_rate, n_fft, path):
    sample_length = len(input_as_array)
    rms = lb.feature.rms(y=input_as_array)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    spectral_flatness = lb.feature.spectral_flatness(y=input_as_array, n_fft=n_fft)
    spectral_bandwidth = lb.feature.spectral_bandwidth(y=input_as_array, sr=sample_rate, n_fft=n_fft)
    spectral_centroid = lb.feature.spectral_centroid(y=input_as_array, sr=sample_rate, n_fft=n_fft)
    zero_crossing_rate = lb.feature.zero_crossing_rate(y=input_as_array)
    return {
        'path': path,
        'filename': os.path.basename(path),
        'sample_rate': sample_rate,
        'sample_length': sample_length,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'spectral_flatness': spectral_flatness,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_centroid': spectral_centroid,
        'zero_crossing_rate': zero_crossing_rate
    }

def process_samples():
    # Load and process data
    start_time = timeit.default_timer()
    data = file_finder_and_sorter(SAMPLE_ROOT_FOLDER)
    list_for_dataframe = []
    success_count = 0
    failure_count = 0
    empty_not_readable_count = 0

    for sample in data:
        try:
            file_name = os.path.basename(sample)    #Extract the filename from the path
            print(f"Processing file: {file_name}")
            audio, sr = load_audio(sample)

            # Skip empty or unreadable samples
            if audio is None or sr is None or len(audio) == 0:
                print(f"{sample} is empty or not readable")
                empty_not_readable_count += 1
                continue    # Skip this sample
            
            preprocessed_audio = preprocess_sample(audio)
            truncated_audio = truncate_if_needed(preprocessed_audio)
            n_fft = adjust_n_fft(truncated_audio)   # Adjust n_fft to be smaller or equal to the input signal length
            features = get_features(input_as_array=truncated_audio, sample_rate=sr, n_fft=n_fft, path=sample)
            list_for_dataframe.append(features)
            success_count += 1
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {sample}: {e}")
            failure_count += 1
    
    df = pd.DataFrame(list_for_dataframe)
    print("\nDataFrame created successfully")

    # Compute the mean and standard deviation for each feature
    df['spectral_flatness_mean'] = df['spectral_flatness'].apply(np.mean)
    df['spectral_flatness_std'] = df['spectral_flatness'].apply(np.std)

    df['spectral_bandwidth_mean'] = df['spectral_bandwidth'].apply(np.mean)
    df['spectral_bandwidth_std'] = df['spectral_bandwidth'].apply(np.std)

    df['spectral_centroid_mean'] = df['spectral_centroid'].apply(np.mean)
    df['spectral_centroid_std'] = df['spectral_centroid'].apply(np.std)

    df['zero_crossing_rate_mean'] = df['zero_crossing_rate'].apply(np.mean)
    df['zero_crossing_rate_std'] = df['zero_crossing_rate'].apply(np.std)

    print('Stds and means successfully calculated\n')

    # Remove the original arrays
    df.drop(columns=['spectral_flatness', 'spectral_bandwidth', 'spectral_centroid', 'zero_crossing_rate'], inplace=True)
    print("Successfully collected ", success_count, " samples")
    print("Failed to collect ", failure_count, " samples")
    print("Empty or not readable samples: ", empty_not_readable_count, "\n")

    #Create final output path
    output_path = os.path.join(CSV_OUTPUT_FOLDER, OUTPUT_CSV_NAME)

    # Save the DataFrame to CSV
    df.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}")
    print(f"Processing took {np.round(timeit.default_timer() - start_time, 3)} seconds\n\n")

if __name__ == "__main__":
    process_samples()