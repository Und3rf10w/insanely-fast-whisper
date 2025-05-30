import torch
from pyannote.audio import Pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn

# AutoConfig is not used in this file, can be removed if not used elsewhere from this import
# from transformers import AutoConfig

from .diarize import (
    post_process_segments_and_transcripts,
    diarize_audio,
    preprocess_inputs,
)


def diarize(args, outputs):
    # Determine the correct value for use_auth_token for pyannote
    # If args.hf_token is the placeholder "no_token" or not provided,
    # set use_auth_token to False, signaling pyannote to attempt loading
    # from cache/local files without online authentication.
    pyannote_token_arg = None  # Default to None
    if args.hf_token and args.hf_token.lower() != "no_token":
        pyannote_token_arg = args.hf_token
        print(
            f"INFO: Using Hugging Face token for pyannote model: {args.diarization_model}"
        )
    else:
        # Important for offline use: use_auth_token=False (or None)
        # tells pyannote not to try to fetch/validate online if model is local.
        pyannote_token_arg = False
        print(
            f"INFO: Attempting to load pyannote model '{args.diarization_model}' from local cache (no Hugging Face token provided/specified as 'no_token')."
        )

    try:
        diarization_pipeline = Pipeline.from_pretrained(
            checkpoint_path=args.diarization_model,
            use_auth_token=pyannote_token_arg,
        )
        diarization_pipeline.to(
            torch.device("mps" if args.device_id == "mps" else f"cuda:{args.device_id}")
        )
    except Exception as e:
        print(
            f"ERROR: Failed to load pyannote diarization pipeline '{args.diarization_model}'. Error: {e}"
        )
        if pyannote_token_arg is False:
            print(
                "INFO: This might be because the model is not available locally or requires a token for the first download."
            )
            print(
                "INFO: If you have a Hugging Face token, try running with --hf-token YOUR_TOKEN."
            )
        raise  # Re-raise the exception to be caught by cli.py for fallback

    # The rest of the function remains the same
    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Pre-processing audio for diarization...", total=None)
        inputs, diarizer_inputs = preprocess_inputs(inputs=args.file_name)

        progress.add_task("[yellow]Running speaker diarization...", total=None)
        # diarize_audio is from .utils.diarize
        segments = diarize_audio(
            diarizer_inputs,
            diarization_pipeline,
            args.num_speakers,
            args.min_speakers,
            args.max_speakers,
        )

        if not segments:
            print("INFO: pyannote.audio returned no speaker segments from the audio.")
            return []  # Return empty list if no segments from diarize_audio

        progress.add_task(
            "[yellow]Post-processing ASR and diarization results...", total=None
        )
        # post_process_segments_and_transcripts is from .utils.diarize
        processed_transcript = post_process_segments_and_transcripts(
            segments,
            outputs["chunks"],
            group_by_speaker=False,  # Assuming you want per-chunk speaker label
        )
        if not processed_transcript:
            print("INFO: No speaker segments could be aligned with the transcript.")
        return processed_transcript
