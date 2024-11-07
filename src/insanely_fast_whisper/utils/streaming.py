import io
import os
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
import speech_recognition as sr
from scipy.io import wavfile
import numpy as np
import torch
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn


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


def setup_microphone(sampling_rate, energy_threshold=400, default_microphone="pulse"):
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

    if os.name == "posix" and default_microphone:
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            if default_microphone in name:
                source = sr.Microphone(sample_rate=sampling_rate, device_index=index)
                break
        else:
            available_mics = "\n".join(sr.Microphone.list_microphone_names())
            raise ValueError(
                f"Microphone '{default_microphone}' not found. Available microphones:\n{available_mics}"
            )
    else:
        source = sr.Microphone(sample_rate=sampling_rate)

    return recorder, source


def process_audio_stream(pipe, diarization_pipeline, args):
    """
    Handles real-time audio streaming, transcription, and diarization.
    """
    sampling_rate = pipe.feature_extractor.sampling_rate
    data_queue = Queue()
    audio_buffer = AudioBuffer()
    phrase_time = None
    last_sample = bytes()

    # Setup microphone and recorder
    recorder, source = setup_microphone(
        sampling_rate,
        energy_threshold=args.energy_threshold,
        default_microphone=getattr(args, "default_microphone", "pulse"),
    )

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    with source:
        recorder.adjust_for_ambient_noise(source)

    recorder.listen_in_background(
        source, record_callback, phrase_time_limit=args.record_timeout
    )

    print("Model loaded and listening. Press Ctrl+C to stop.\n")

    transcription = []
    try:
        while True:
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
                audio_buffer.add(audio_array.astype(np.float32) / 32768.0)

                # Process with both pipelines
                with Progress(
                    TextColumn("🤗 [progress.description]{task.description}"),
                    BarColumn(style="yellow1", pulse_style="white"),
                    TimeElapsedColumn(),
                ) as progress:
                    progress.add_task("[yellow]Processing...", total=None)

                    # Get transcription
                    outputs = pipe(
                        audio_array,
                        chunk_length_s=30,
                        batch_size=args.batch_size,
                        generate_kwargs={"task": args.task, "language": args.language},
                        return_timestamps=True,
                    )

                    if diarization_pipeline is not None:
                        # Process diarization on buffered audio
                        diarization = diarization_pipeline(
                            {
                                "waveform": torch.from_numpy(
                                    audio_buffer.get()
                                ).unsqueeze(0),
                                "sample_rate": sampling_rate,
                            }
                        )

                        # Process segments
                        segments = []
                        for segment, track, label in diarization.itertracks(
                            yield_label=True
                        ):
                            segments.append(f"[{label}]: {outputs['text'].strip()}")

                        text = "\n".join(segments)
                    else:
                        text = outputs["text"].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Clear the console and show updated transcription
                os.system("cls" if os.name == "nt" else "clear")
                for line in transcription:
                    print(line)
                print("", end="", flush=True)

                sleep(0.25)

    except KeyboardInterrupt:
        print("\nTranscription ended.")
        return transcription
