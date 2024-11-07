import argparse
import io
import os
import torch
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
import speech_recognition as sr
from scipy.io import wavfile
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from transformers import pipeline
from pyannote.audio import Pipeline
import numpy as np


class AudioBuffer:
    def __init__(self, max_size=48000 * 30):  # 30 seconds at 48kHz
        self.buffer = np.array([], dtype=np.float32)
        self.max_size = max_size

    def add(self, audio_data):
        self.buffer = np.append(self.buffer, audio_data)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size :]

    def get(self):
        return self.buffer.copy()


def main():
    parser = argparse.ArgumentParser()
    # [Previous arguments remain the same]
    parser.add_argument(
        "--hf-token",
        required=True,
        type=str,
        help="HuggingFace token for pyannote.audio",
    )
    args = parser.parse_args()

    # Initialize buffers and queues
    data_queue = Queue()
    audio_buffer = AudioBuffer()
    phrase_time = None
    last_sample = bytes()

    # Initialize recorder
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Initialize both pipelines
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs=(
            {"attn_implementation": "flash_attention_2"}
            if args.flash
            else {"attn_implementation": "sdpa"}
        ),
    )

    diarization_pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=args.hf_token,
    )
    diarization_pipe.to(
        torch.device("mps" if args.device_id == "mps" else f"cuda:{args.device_id}")
    )

    sampling_rate = asr_pipe.feature_extractor.sampling_rate

    # Set up microphone source
    if "linux" in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(
                        sample_rate=sampling_rate, device_index=index
                    )
                    break
    else:
        source = sr.Microphone(sample_rate=sampling_rate)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    with source:
        recorder.adjust_for_ambient_noise(source)

    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=args.record_timeout
    )

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(
                    seconds=args.phrase_timeout
                ):
                    last_sample = bytes()
                    phrase_complete = True
                phrase_time = now

                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                audio_data = sr.AudioData(
                    last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH
                )
                wav_data = io.BytesIO(audio_data.get_wav_data())
                sample_rate, audio_array = wavfile.read(wav_data)

                # Add to buffer for diarization
                audio_buffer.add(
                    audio_array.astype(np.float32) / 32768.0
                )  # Normalize to [-1, 1]

                # Process with both pipelines
                with Progress(
                    TextColumn("🤗 Processing..."), BarColumn(), TimeElapsedColumn()
                ) as progress:
                    task = progress.add_task("Transcribing...", total=None)

                    # Get transcription
                    asr_output = asr_pipe(
                        audio_array,
                        chunk_length_s=30,
                        batch_size=args.batch_size,
                        generate_kwargs={"task": args.task, "language": args.language},
                        return_timestamps=True,
                    )

                    # Process diarization on buffered audio
                    diarization = diarization_pipe(
                        {
                            "waveform": torch.from_numpy(audio_buffer.get()).unsqueeze(
                                0
                            ),
                            "sample_rate": sampling_rate,
                        }
                    )

                # Combine results
                text = asr_output["text"].strip()
                segments = []
                for segment, track, label in diarization.itertracks(yield_label=True):
                    segments.append(f"[{label}]: {text}")

                # Update display
                os.system("cls" if os.name == "nt" else "clear")
                for segment in segments:
                    print(segment)
                print("", end="", flush=True)

                sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\nTranscription ended.")


if __name__ == "__main__":
    main()
