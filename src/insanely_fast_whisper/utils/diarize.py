import requests
import torch
import numpy as np
from torchaudio import functional as F
from transformers.pipelines.audio_utils import ffmpeg_read
import sys


# Code lifted from https://github.com/huggingface/speechbox/blob/main/src/speechbox/diarize.py
# and from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py


def preprocess_inputs(inputs):
    if isinstance(inputs, str):
        if inputs.startswith("http://") or inputs.startswith("https://"):
            # We need to actually check for a real protocol, otherwise it's impossible to use a local file
            # like http_huggingface_co.png
            inputs = requests.get(inputs).content
        else:
            with open(inputs, "rb") as f:
                inputs = f.read()

    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, 16000)

    if isinstance(inputs, dict):
        # Accepting `"array"` which is the key defined in `datasets` for better integration
        if not ("sampling_rate" in inputs and ("raw" in inputs or "array" in inputs)):
            raise ValueError(
                "When passing a dictionary to ASRDiarizePipeline, the dict needs to contain a "
                '"raw" key containing the numpy array representing the audio and a "sampling_rate" key, '
                "containing the sampling_rate associated with that array"
            )

        _inputs = inputs.pop("raw", None)
        if _inputs is None:
            # Remove path which will not be used from `datasets`.
            inputs.pop("path", None)
            _inputs = inputs.pop("array", None)
        in_sampling_rate = inputs.pop("sampling_rate")
        inputs = _inputs
        if in_sampling_rate != 16000:
            inputs = F.resample(
                torch.from_numpy(inputs), in_sampling_rate, 16000
            ).numpy()

    if not isinstance(inputs, np.ndarray):
        raise ValueError(f"We expect a numpy ndarray as input, got `{type(inputs)}`")
    if len(inputs.shape) != 1:
        raise ValueError(
            "We expect a single channel audio input for ASRDiarizePipeline"
        )

    # diarization model expects float32 torch tensor of shape `(channels, seq_len)`
    diarizer_inputs = torch.from_numpy(inputs).float()
    diarizer_inputs = diarizer_inputs.unsqueeze(0)

    return inputs, diarizer_inputs


def diarize_audio(
    diarizer_inputs, diarization_pipeline, num_speakers, min_speakers, max_speakers
):
    diarization = diarization_pipeline(
        {"waveform": diarizer_inputs, "sample_rate": 16000},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "segment": {"start": segment.start, "end": segment.end},
                "track": track,
                "label": label,
            }
        )

    # Add this check:
    if not segments:
        print("Warning: Diarization pipeline returned no segments.")
        return []  # Return an empty list if no segments were found

    # The rest of the function remains the same:
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        # The condition `and i < len(segments)` is redundant here as `i` is always less than `len(segments)` in this loop.
        # It can be safely removed, but leaving it doesn't harm.
        if (
            cur_segment["label"] != prev_segment["label"]
        ):  # Removed redundant `and i < len(segments)`
            new_segments.append(
                {
                    "segment": {
                        "start": prev_segment["segment"]["start"],
                        "end": cur_segment["segment"]["start"],
                    },
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = cur_segment  # Corrected from prev_segment = segments[i] for clarity, though functionally similar here

    new_segments.append(
        {
            "segment": {
                "start": prev_segment["segment"]["start"],
                "end": cur_segment["segment"]["end"],
            },
            "speaker": prev_segment["label"],
        }
    )

    return new_segments


def post_process_segments_and_transcripts(
    new_segments, transcript, group_by_speaker
) -> list:
    # Add this check at the beginning:
    if not transcript:  # If ASR produced no transcribed chunks
        # If there are speaker segments but no text, it's ambiguous what to return.
        # Returning an empty list is safest as no combined segments can be formed.
        return []

    if not new_segments:  # If diarization produced no speaker segments
        # If there's a transcript but no speaker segments, we can't assign speakers.
        # Depending on desired behavior, one might return the transcript with a default "UNKNOWN" speaker
        # or, as done here by implication if segmented_preds remains empty, return an empty list.
        # For consistency with the function's goal (producing speaker-segmented transcript),
        # if no speaker segments, then no such output can be made.
        # However, the calling code in cli.py uses build_result which might expect original chunks.
        # For now, let's ensure it returns empty if new_segments is empty, and cli.py handles it.
        # The current logic will result in segmented_preds = [] if new_segments is empty, which is fine.
        pass

    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array(
        [
            (
                chunk["timestamp"][-1]
                if chunk["timestamp"][-1] is not None
                else sys.float_info.max
            )
            for chunk in transcript
        ]
    )
    # This check is important if transcript might be non-empty but all timestamps are None
    # However, np.argmin([]) is the main issue if transcript itself was empty.
    # If end_timestamps is empty due to all chunk timestamps being None, but transcript is not empty,
    # np.argmin will also raise ValueError. This is an edge case.
    # Assuming Whisper chunks usually have timestamps.

    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join(
                        [chunk["text"] for chunk in transcript[: upto_idx + 1]]
                    ),
                    "timestamp": (
                        transcript[0]["timestamp"][0],
                        transcript[upto_idx]["timestamp"][1],
                    ),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

        if len(end_timestamps) == 0:
            break

    return segmented_preds
