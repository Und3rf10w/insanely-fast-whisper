import os
import numpy as np
import pyaudio
import wave
import threading
from queue import Queue
from datetime import datetime, timedelta
from time import sleep
import torch
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn


class AudioStreamer:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.audio_queue = Queue()
        self.is_recording = False

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start_streaming(self):
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.callback,
        )
        self.is_recording = True
        self.stream.start_stream()

    def stop_streaming(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


def process_audio_stream(pipe, diarization_pipeline, args):
    """
    Handles real-time audio streaming, transcription, and optional diarization.
    """
    # Set up the same parameters as non-streaming mode
    ts = "word" if args.timestamp == "word" else True
    language = None if args.language == "None" else args.language

    # Construct generate_kwargs the same way as non-streaming mode
    generate_kwargs = {"task": args.task}
    if language is not None:
        generate_kwargs["language"] = language

    # Remove task for English-only models
    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    audio_streamer = AudioStreamer(sample_rate=pipe.feature_extractor.sampling_rate)
    transcription = []
    audio_buffer = []
    last_process_time = datetime.utcnow()

    try:
        audio_streamer.start_streaming()
        print("Model loaded and listening. Press Ctrl+C to stop.\n")

        while True:
            now = datetime.utcnow()

            # Collect audio data
            while not audio_streamer.audio_queue.empty():
                audio_data = audio_streamer.audio_queue.get()
                audio_buffer.extend(audio_data)

            # Process if we have enough data or enough time has passed
            if (
                len(audio_buffer)
                >= pipe.feature_extractor.sampling_rate * args.record_timeout
                or (now - last_process_time).total_seconds() >= args.phrase_timeout
            ):

                if len(audio_buffer) > 0:
                    audio_array = np.array(audio_buffer)

                    with Progress(
                        TextColumn("🤗 [progress.description]{task.description}"),
                        BarColumn(style="yellow1", pulse_style="white"),
                        TimeElapsedColumn(),
                    ) as progress:
                        # Transcription
                        progress.add_task("[yellow]Transcribing...", total=None)
                        outputs = pipe(
                            audio_array,
                            chunk_length_s=30,
                            batch_size=args.batch_size,
                            generate_kwargs=generate_kwargs,
                            return_timestamps=ts,
                        )

                        text = outputs["text"].strip()

                        # Optional diarization
                        if diarization_pipeline is not None:
                            progress.add_task("[yellow]Diarizing...", total=None)
                            diarization = diarization_pipeline(
                                {
                                    "waveform": torch.from_numpy(audio_array).unsqueeze(
                                        0
                                    ),
                                    "sample_rate": audio_streamer.sample_rate,
                                },
                                num_speakers=args.num_speakers,
                                min_speakers=args.min_speakers,
                                max_speakers=args.max_speakers,
                            )

                            # Process segments with speaker labels
                            segments = []
                            for segment, track, label in diarization.itertracks(
                                yield_label=True
                            ):
                                segments.append(f"[{label}]: {text}")
                            text = "\n".join(segments)

                    transcription.append(text)
                    audio_buffer = []
                    last_process_time = now

                    # Update display
                    os.system("cls" if os.name == "nt" else "clear")
                    for line in transcription:
                        print(line)
                    print("", end="", flush=True)

            sleep(0.1)

    except KeyboardInterrupt:
        audio_streamer.stop_streaming()
        print("\nTranscription ended.")
        return transcription
