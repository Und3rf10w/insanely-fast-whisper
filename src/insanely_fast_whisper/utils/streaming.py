import os
import numpy as np
import pyaudio
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from time import sleep
import torch
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn


class AudioStreamer:
    def __init__(
        self,
        sample_rate=16000,
        chunk_size=1024,
        channels=1,
        record_timeout=2,
        phrase_timeout=3,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.buffer_queue = Queue()  # Queue for chunks ready for processing
        self.current_buffer = []  # Current recording buffer
        self.buffer_lock = threading.Lock()
        self.is_recording = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.last_process_time = datetime.utcnow()

    def callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            with self.buffer_lock:
                self.current_buffer.extend(audio_data)

                # Check if we have enough data for a chunk
                current_time = datetime.utcnow()
                if (
                    len(self.current_buffer) >= self.sample_rate * self.record_timeout
                    or (current_time - self.last_process_time).total_seconds()
                    >= self.phrase_timeout
                ):
                    if len(self.current_buffer) > 0:
                        # Put the current buffer in the processing queue
                        self.buffer_queue.put(np.array(self.current_buffer))
                        self.current_buffer = []  # Start a new buffer
                        self.last_process_time = current_time

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
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


def process_audio_chunks(
    audio_streamer, pipe, diarization_pipeline, args, transcription
):
    """
    Process audio chunks in a separate thread.
    """
    ts = "word" if args.timestamp == "word" else True
    language = None if args.language == "None" else args.language

    generate_kwargs = {"task": args.task}
    if language is not None:
        generate_kwargs["language"] = language

    if args.model_name.split(".")[-1] == "en":
        generate_kwargs.pop("task")

    while not audio_streamer.stop_event.is_set():
        try:
            # Get the next chunk from the queue with timeout
            audio_chunk = audio_streamer.buffer_queue.get(timeout=0.5)

            with Progress(
                TextColumn("🤗 [progress.description]{task.description}"),
                BarColumn(style="yellow1", pulse_style="white"),
                TimeElapsedColumn(),
            ) as progress:
                # Transcription
                progress.add_task("[yellow]Transcribing...", total=None)
                outputs = pipe(
                    audio_chunk,
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
                            "waveform": torch.from_numpy(audio_chunk).unsqueeze(0),
                            "sample_rate": audio_streamer.sample_rate,
                        },
                        num_speakers=args.num_speakers,
                        min_speakers=args.min_speakers,
                        max_speakers=args.max_speakers,
                    )

                    segments = []
                    for segment, track, label in diarization.itertracks(
                        yield_label=True
                    ):
                        segments.append(f"[{label}]: {text}")
                    text = "\n".join(segments)

            if text.strip():  # Only add non-empty transcriptions
                transcription.append(text)

                # Update display
                os.system("cls" if os.name == "nt" else "clear")
                for line in transcription:
                    print(line)
                print("", end="", flush=True)

        except Empty:
            continue  # No chunks available, continue waiting

        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            continue


def process_audio_stream(pipe, diarization_pipeline, args):
    """
    Handles real-time audio streaming, transcription, and optional diarization.
    """
    audio_streamer = AudioStreamer(
        sample_rate=pipe.feature_extractor.sampling_rate,
        record_timeout=args.record_timeout,
        phrase_timeout=args.phrase_timeout,
    )
    transcription = []

    try:
        # Start the audio streaming
        audio_streamer.start_streaming()

        # Start the processing thread
        audio_streamer.processing_thread = threading.Thread(
            target=process_audio_chunks,
            args=(audio_streamer, pipe, diarization_pipeline, args, transcription),
        )
        audio_streamer.processing_thread.start()

        print("Model loaded and listening. Press Ctrl+C to stop.\n")

        # Keep the main thread alive until interrupted
        while not audio_streamer.stop_event.is_set():
            sleep(0.1)

    except KeyboardInterrupt:
        audio_streamer.stop_streaming()
        print("\nTranscription ended.")
        return transcription
