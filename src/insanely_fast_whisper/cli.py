import json
import argparse
from transformers import pipeline
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
import torch

# Assuming these imports are correctly pointing to your utility functions
from .utils.diarization_pipeline import diarize
from .utils.result import build_result

# The streaming import is conditional later, so keep it accessible
# from .utils.streaming import process_audio_stream


# Argument parser setup (as provided in the original file)
parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=False,
    type=str,
    help="Path or URL to the audio file to be transcribed.",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "mps" for Macs with Apple Silicon. (default: "0")',
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",  # Keep as string "None" for easier checking later
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,  # For boolean flags, action="store_true" is better
    default=False,  # If type=bool, it expects True/False. If using action, it's a flag.
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
# Consider changing --flash to action="store_true" if it's meant to be a flag without an argument
# If it's type=bool, the user must pass --flash True or --flash False.
# For this implementation, I'll assume the current type=bool is intended.

parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)
parser.add_argument(
    "--hf-token",
    required=False,
    default="no_token",  # Default indicating no token provided
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips. (default: 'no_token')",
)
parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",  # Default model
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization-3.1)",
)
parser.add_argument(
    "--num-speakers",
    required=False,
    default=None,
    type=int,
    help="Specifies the exact number of speakers present in the audio file. (default: None)",
)
parser.add_argument(
    "--min-speakers",
    required=False,
    default=None,
    type=int,
    help="Sets the minimum number of speakers for diarization. (default: None)",
)
parser.add_argument(
    "--max-speakers",
    required=False,
    default=None,
    type=int,
    help="Defines the maximum number of speakers for diarization. (default: None)",
)
parser.add_argument(
    "--live-transcribe",
    action="store_true",  # This is a boolean flag
    help="Enable live transcription from microphone input.",
)
parser.add_argument(
    "--energy-threshold",  # For live transcription
    default=400,
    help="Energy level for mic to detect (for live transcription).",
    type=int,
)
parser.add_argument(
    "--record-timeout",  # For live transcription
    default=2,
    help="How real time the recording is in seconds (for live transcription).",
    type=float,
)
parser.add_argument(
    "--phrase-timeout",  # For live transcription
    default=3,
    help="How much empty space between recordings before we consider it a new line (for live transcription).",
    type=float,
)


def main():
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error(
            "--num-speakers cannot be used together with --min-speakers or --max-speakers."
        )
    if args.num_speakers is not None and args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")
    if args.min_speakers is not None and args.min_speakers < 1:
        parser.error("--min-speakers must be at least 1.")
    if args.max_speakers is not None and args.max_speakers < 1:
        parser.error("--max-speakers must be at least 1.")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot be greater than --max-speakers.")

    if args.live_transcribe and args.file_name:
        parser.error(
            "--file-name should not be specified when using --live-transcribe."
        )
    if not args.live_transcribe and not args.file_name:
        parser.error("--file-name is required when not using --live-transcribe.")

    # --- ASR Pipeline Initialization ---
    attn_implementation = "sdpa"  # Scaled Dot Product Attention, good default
    if args.flash:
        attn_implementation = "flash_attention_2"
        print("INFO: Using Flash Attention 2. Ensure it's installed correctly.")

    model_kwargs = {"attn_implementation": attn_implementation}

    print(
        f"INFO: Initializing ASR pipeline with model: {args.model_name} on device: {args.device_id} using {attn_implementation}..."
    )
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        torch_dtype=torch.float16,  # Standard for Whisper models
        device="mps" if args.device_id == "mps" else f"cuda:{args.device_id}",
        model_kwargs=model_kwargs,
    )
    print("INFO: ASR pipeline initialized.")

    if args.device_id == "mps":
        torch.mps.empty_cache()
    # The .to_bettertransformer() was commented out, keeping it that way unless specifically needed.
    # It's generally for older Hugging Face versions or specific speedups not always compatible with all models/features.
    # if not args.flash and args.device_id != "mps": # BetterTransformer not compatible with MPS or Flash Attn
    #    try:
    #        pipe.model = pipe.model.to_bettertransformer()
    #        print("INFO: Applied BetterTransformer optimization.")
    #    except Exception as e:
    #        print(f"WARNING: Failed to apply BetterTransformer: {e}")

    # --- Transcription Task Configuration ---
    generate_kwargs = {"task": args.task}
    if args.language and args.language.lower() != "none":
        generate_kwargs["language"] = args.language
        print(f"INFO: Language set to: {args.language}")
    else:
        print("INFO: Language will be auto-detected by Whisper.")

    # For English-only models, "task" and "language" might not be needed or cause issues
    if ".en" in args.model_name:  # A common convention for English-only Whisper models
        if "language" in generate_kwargs:
            # print("INFO: English-only model detected, removing explicit language from generate_kwargs.")
            # generate_kwargs.pop("language") # Whisper might handle this fine, or error if language is passed.
            pass  # Let Whisper decide if it needs it.
        if "task" in generate_kwargs and generate_kwargs["task"] == "translate":
            print(
                "WARNING: Translation task requested with an English-only model. This might not work as expected."
            )
        # For English-only models, task is implicitly "transcribe".
        # generate_kwargs.pop("task", None) # Some Whisper versions might not need task for .en models

    return_timestamps_option = "word" if args.timestamp == "word" else True
    print(
        f"INFO: Timestamps requested: {'word-level' if return_timestamps_option == 'word' else 'chunk-level'}"
    )

    # --- Main Processing Logic ---
    if args.live_transcribe:
        # --- Live Transcription ---
        print("INFO: Live transcription mode enabled.")
        from .utils.streaming import (
            process_audio_stream,
        )  # Import here as it's specific to live mode

        diarization_pipeline_live = None
        # Determine if diarization is requested for live mode
        live_diarization_requested = False
        if args.diarization_model:  # If a model is specified
            live_diarization_requested = True
        if (
            args.num_speakers is not None
            or args.min_speakers is not None
            or args.max_speakers is not None
        ):
            live_diarization_requested = True

        if live_diarization_requested:
            print(
                f"INFO: Diarization requested for live transcription with model: {args.diarization_model}"
            )
            pyannote_token_arg_live = None
            if args.hf_token and args.hf_token.lower() != "no_token":
                pyannote_token_arg_live = args.hf_token
                print("INFO: Using Hugging Face token for pyannote model (live).")
            else:
                pyannote_token_arg_live = False  # For offline/cached live diarization
                print(
                    "INFO: Attempting to load pyannote model from local cache for live diarization (no token)."
                )

            try:
                from pyannote.audio import (
                    Pipeline as PyannotePipeline,
                )  # Alias to avoid conflict if any

                diarization_pipeline_live = PyannotePipeline.from_pretrained(
                    checkpoint_path=args.diarization_model,
                    use_auth_token=pyannote_token_arg_live,
                )
                diarization_pipeline_live.to(
                    torch.device(
                        "mps" if args.device_id == "mps" else f"cuda:{args.device_id}"
                    )
                )
                print(
                    "INFO: Pyannote diarization pipeline initialized for live transcription."
                )
            except Exception as e:
                print(
                    f"ERROR: Failed to load pyannote diarization pipeline for live mode: {e}. Diarization will be skipped."
                )
                diarization_pipeline_live = None  # Ensure it's None if loading failed

        transcription_lines = process_audio_stream(
            pipe, diarization_pipeline_live, args
        )  # Pass potentially None pipeline

        if args.transcript_path:
            print(f"INFO: Saving live transcription to: {args.transcript_path}")
            with open(args.transcript_path, "w", encoding="utf8") as fp:
                # For live, the result structure might be simpler, just joined text or structured if process_audio_stream provides it
                # Assuming process_audio_stream returns a list of strings for now.
                # The original code built a result with empty chunks/speakers for live.
                live_output_text = (
                    "\n".join(transcription_lines) if transcription_lines else ""
                )
                result = build_result(
                    [], {"chunks": [], "text": live_output_text}
                )  # Match original structure
                json.dump(result, fp, ensure_ascii=False)
            print(f"INFO: Live transcription saved.")

    else:
        # --- File-based Transcription ---
        print(f"INFO: Processing file: {args.file_name}")
        with Progress(
            TextColumn("🤗 [progress.description]{task.description}"),
            BarColumn(style="yellow1", pulse_style="white"),
            TimeElapsedColumn(),
        ) as progress:
            progress.add_task("[yellow]Transcribing audio file...", total=None)
            outputs = pipe(
                args.file_name,  # URL or local path
                chunk_length_s=30,  # Standard for Whisper
                batch_size=args.batch_size,
                generate_kwargs=generate_kwargs,
                return_timestamps=return_timestamps_option,
            )
        print("INFO: Transcription complete.")
        # outputs will contain {"text": "...", "chunks": [...]}

        speakers_transcript = []  # Default to empty list
        print_msg_suffix = "Your file has been transcribed."  # Default message part

        # Determine if diarization should be performed based on arguments
        should_perform_diarization = False
        if args.diarization_model:  # If a diarization model is specified (even default)
            should_perform_diarization = True
        if (
            args.num_speakers is not None
            or args.min_speakers is not None
            or args.max_speakers is not None
        ):
            should_perform_diarization = (
                True  # Speaker constraints also imply diarization intent
            )

        if should_perform_diarization:
            print(
                f"INFO: Attempting speaker diarization with model: {args.diarization_model}..."
            )
            try:
                # The `diarize` function (from .utils.diarization_pipeline)
                # is responsible for initializing pyannote correctly based on args.hf_token
                # and performing the diarization and merging.
                speakers_transcript = diarize(args, outputs)

                if (
                    not speakers_transcript
                ):  # diarize returns empty list on failure or no segments
                    print(
                        "INFO: Diarization process completed but resulted in no distinct speaker segments for the transcript."
                    )
                    print_msg_suffix = "Your file has been transcribed (diarization yielded no speaker segments)."
                else:
                    print("INFO: Diarization successful, speaker segments identified.")
                    print_msg_suffix = (
                        "Your file has been transcribed & speaker segmented."
                    )
            except Exception as e:
                print(f"ERROR: Diarization failed with an error: {e}")
                print(
                    "INFO: Falling back to transcription without speaker diarization."
                )
                # speakers_transcript remains []
                print_msg_suffix = (
                    "Your file has been transcribed (diarization failed)."
                )
        else:
            # This case might occur if user explicitly sets --diarization_model to None or an empty string,
            # and no speaker count args are given. Or if we change logic to not assume default model means intent.
            # For now, with a default diarization_model, this branch is less likely.
            print(
                "INFO: Diarization not performed (not requested or model not specified)."
            )
            print_msg_suffix = (
                "Your file has been transcribed (diarization not performed)."
            )

        # Save the result
        print(f"INFO: Saving transcription to: {args.transcript_path}")
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            # build_result will use the (potentially empty) speakers_transcript
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ {print_msg_suffix} Check it out here 👉 {args.transcript_path}"
        )


if __name__ == "__main__":
    main()
