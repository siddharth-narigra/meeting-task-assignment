"""
transcriber.py - Speech-to-Text Module using OpenAI Whisper

This module handles the conversion of audio files to text using the
local Whisper model. It's the first step in our pipeline.

How it works:
1. Load the Whisper model (we use "base" for good speed/accuracy balance)
2. Feed it an audio file (wav, mp3, or m4a)
3. Get back the transcribed text

Why Whisper?
- Works locally (no internet/API needed after download)
- Supports multiple languages
- Handles various audio qualities well
- Free to use
"""

import whisper  # type: ignore
import os


class Transcriber:
    """
    Handles audio-to-text transcription using Whisper.
    
    The model is loaded once when the class is instantiated,
    then reused for all transcriptions (more efficient).
    
    Available models (from fastest to most accurate):
    - "tiny"   : ~1GB RAM, fastest, good for quick tests
    - "base"   : ~1GB RAM, good balance (WE USE THIS)
    - "small"  : ~2GB RAM, better accuracy
    - "medium" : ~5GB RAM, even better
    - "large"  : ~10GB RAM, best accuracy but slow on CPU
    """
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the transcriber with a specific Whisper model.
        
        Args:
            model_name: Which Whisper model to use. Default is "base".
                       On first run, the model will be downloaded (~140MB for base)
        
        Example:
            transcriber = Transcriber()  # Uses "base" model
            transcriber = Transcriber("small")  # Uses "small" model for better accuracy
        """
        print(f"Loading Whisper '{model_name}' model...")
        print("(This may take a moment on first run as the model downloads)")
        
        # Load the model - this downloads it if not already cached
        # Models are cached in ~/.cache/whisper/
        self.model = whisper.load_model(model_name)
        
        print(f"Whisper '{model_name}' model loaded successfully!")
    
    def transcribe(self, audio_path: str) -> str:
        """
        Convert an audio file to text.
        
        Args:
            audio_path: Path to the audio file (wav, mp3, or m4a)
            
        Returns:
            The transcribed text as a string
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            
        Example:
            text = transcriber.transcribe("meeting.wav")
            print(text)
            # Output: "Hi everyone, let's discuss this week's priorities..."
        """
        # Check if file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Get the file extension to show what we're processing
        file_ext = os.path.splitext(audio_path)[1].lower()
        print(f"Transcribing {file_ext} file: {audio_path}")
        print("(This may take a while depending on audio length and your CPU)")
        
        # Transcribe the audio
        # fp16=False is important for CPU - it forces 32-bit precision
        # which is slower but works on CPUs without float16 support
        result = self.model.transcribe(audio_path, fp16=False)
        
        # The result is a dictionary with several keys:
        # - "text": The full transcription (what we want)
        # - "segments": Timestamped chunks of text
        # - "language": Detected language
        transcript = result["text"].strip()
        
        print(f"Transcription complete! ({len(transcript)} characters)")
        
        return transcript


# This allows testing the module directly
if __name__ == "__main__":
    # Example usage - you can test with any audio file
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file>")
        print("Example: python transcriber.py meeting.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Create transcriber and process the file
    t = Transcriber()
    text = t.transcribe(audio_file)
    
    print("\n" + "="*50)
    print("TRANSCRIPT:")
    print("="*50)
    print(text)

