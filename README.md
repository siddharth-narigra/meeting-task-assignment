# Meeting Task Assignment System

Automatically extract and assign tasks from meeting audio recordings using speech-to-text and custom NLP logic.

## Features

- **Speech-to-Text**: Uses OpenAI Whisper (local model, no API key needed)
- **Task Extraction**: Custom NLP rules to identify tasks, deadlines, and priorities
- **Smart Assignment**: Skill-based matching to assign tasks to the right team members
- **Dependency Detection**: Identifies task dependencies mentioned in the meeting

## Project Structure

```
task/
├── main.py              # CLI entry point
├── transcriber.py       # Whisper speech-to-text wrapper
├── task_extractor.py    # Custom task identification logic
├── task_assigner.py     # Skill-matching assignment logic
├── models.py            # Data classes (Task, TeamMember)
├── sample_team.json     # Example team configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system

### Installing FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# OR using winget
winget install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy language model:**
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
python main.py --audio <audio_file> --team <team_json> --output <output_json>
```

### Example

```bash
python main.py --audio meeting.wav --team sample_team.json --output tasks.json
```

### Command Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--audio` | `-a` | Yes | Path to audio file (wav, mp3, m4a) |
| `--team` | `-t` | Yes | Path to team members JSON file |
| `--output` | `-o` | No | Output JSON path (default: tasks_output.json) |
| `--model` | `-m` | No | Whisper model size: tiny, base, small, medium, large (default: base) |

### Whisper Model Sizes

| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| tiny | ~1GB | Fastest | Lower | Quick tests |
| base | ~1GB | Fast | Good | **Default - Best balance** |
| small | ~2GB | Medium | Better | Better accuracy needed |
| medium | ~5GB | Slow | High | High-quality transcription |
| large | ~10GB | Slowest | Best | Maximum accuracy |

## Team JSON Format

Create a JSON file with your team members:

```json
{
  "team_members": [
    {
      "name": "Sakshi",
      "role": "Frontend Developer",
      "skills": ["React", "JavaScript", "UI bugs"]
    },
    {
      "name": "Mohit",
      "role": "Backend Engineer",
      "skills": ["Database", "APIs", "Performance optimization"]
    }
  ]
}
```

## Output JSON Format

The system outputs tasks in this format:

```json
{
  "tasks": [
    {
      "id": 1,
      "task": "Fix critical login bug",
      "description": "We need to fix the critical login bug...",
      "assigned_to": "Sakshi",
      "deadline": "Tomorrow Evening",
      "priority": "Critical",
      "dependencies": [],
      "reason": "Skills: JavaScript, UI bugs"
    }
  ]
}
```

## How It Works

### 1. Speech-to-Text (transcriber.py)
- Uses OpenAI Whisper to convert audio to text
- Runs locally (no internet/API needed after model download)
- Model is cached after first download (~140MB for base)

### 2. Task Extraction (task_extractor.py)
Custom NLP logic that:
- Segments text into sentences using spaCy
- Detects task indicators: "need to", "should", "has to", etc.
- Extracts deadlines: "tomorrow", "by Friday", "next week"
- Identifies priorities: "critical", "urgent", "can wait"
- Finds dependencies: "depends on", "after X is done"
- Detects person mentions for direct assignment

### 3. Task Assignment (task_assigner.py)
Custom matching algorithm:
- **Direct Mention**: If someone is named, assign to them
- **Skill Matching**: Scores team members by skill overlap with task
- **Role Matching**: Maps task types to relevant roles

## Examples

### Running with a Sample Meeting

```bash
# Using the provided sample team
python main.py -a meeting_recording.wav -t sample_team.json -o my_tasks.json
```

### Using a Different Whisper Model

```bash
# Use 'small' model for better accuracy (slower)
python main.py -a meeting.mp3 -t team.json -o tasks.json --model small

# Use 'tiny' model for quick testing (less accurate)
python main.py -a meeting.mp3 -t team.json --model tiny
```

## Troubleshooting

### "FFmpeg not found" error
Make sure FFmpeg is installed and in your system PATH:
```bash
ffmpeg -version
```

### "No tasks identified" warning
- Check audio quality (clear speech works better)
- Ensure the meeting discusses actionable tasks
- Try a larger Whisper model for better transcription

### Slow transcription
- The first run downloads the model (~140MB)
- Use `--model tiny` for faster (but less accurate) results
- CPU processing is slower than GPU

## License

MIT License

