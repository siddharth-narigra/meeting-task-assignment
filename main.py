"""
main.py - CLI Entry Point for Meeting Task Assignment System

This is the main entry point that orchestrates the entire pipeline:
1. Load team member data from JSON
2. Transcribe audio using Whisper
3. Extract tasks using custom NLP
4. Assign tasks using skill matching
5. Output results to JSON

Usage:
    python main.py --audio meeting.wav --team team.json --output tasks.json
    
    Or with short flags:
    python main.py -a meeting.wav -t team.json -o tasks.json
"""

import argparse
import json
import os
import sys
from typing import List

from models import Task, TeamMember
from transcriber import Transcriber
from task_extractor import TaskExtractor
from task_assigner import TaskAssigner


def load_team_members(json_path: str) -> List[TeamMember]:
    """
    Load team members from a JSON file.
    
    Expected JSON format:
    {
        "team_members": [
            {"name": "Sakshi", "role": "Frontend Developer", "skills": ["React", "JavaScript"]}
        ]
    }
    
    Args:
        json_path: Path to the team JSON file
        
    Returns:
        List of TeamMember objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Team file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'team_members' not in data:
        raise ValueError("JSON must contain 'team_members' key")
    
    team_members = []
    for member_data in data['team_members']:
        member = TeamMember(
            name=member_data['name'],
            role=member_data['role'],
            skills=member_data.get('skills', [])
        )
        team_members.append(member)
    
    return team_members


def save_tasks_to_json(tasks: List[Task], output_path: str) -> None:
    """
    Save extracted tasks to a JSON file.
    
    Output format:
    {
        "tasks": [
            {
                "id": 1,
                "task": "Fix login bug",
                "description": "...",
                "assigned_to": "Sakshi",
                ...
            }
        ]
    }
    
    Args:
        tasks: List of Task objects to save
        output_path: Path where to save the JSON
    """
    output_data = {
        "tasks": [task.to_dict() for task in tasks]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def print_tasks_table(tasks: List[Task]) -> None:
    """
    Print tasks in a formatted table to console.
    
    This gives a nice visual summary of the extracted tasks.
    
    Args:
        tasks: List of Task objects to display
    """
    print("\n" + "="*100)
    print("MEETING TASK ASSIGNMENTS")
    print("="*100)
    
    # Header
    print(f"{'#':<3} {'Task':<35} {'Assigned To':<15} {'Deadline':<20} {'Priority':<10}")
    print("-"*100)
    
    # Rows
    for task in tasks:
        task_name = task.task[:32] + "..." if len(task.task) > 35 else task.task
        deadline = task.deadline or "Not specified"
        assigned = task.assigned_to or "Unassigned"
        
        print(f"{task.id:<3} {task_name:<35} {assigned:<15} {deadline:<20} {task.priority:<10}")
    
    print("-"*100)
    print(f"Total tasks: {len(tasks)}")
    print("="*100)
    
    # Print detailed view with reasons
    print("\nDETAILED ASSIGNMENTS:")
    print("-"*60)
    for task in tasks:
        print(f"\n#{task.id}: {task.task}")
        print(f"   Description: {task.description}")
        print(f"   Assigned to: {task.assigned_to or 'Unassigned'}")
        print(f"   Deadline: {task.deadline or 'Not specified'}")
        print(f"   Priority: {task.priority}")
        if task.dependencies:
            print(f"   Dependencies: Task(s) #{task.dependencies}")
        print(f"   Reason: {task.reason or 'N/A'}")


def main():
    """
    Main function - entry point for the CLI.
    
    Parses arguments and runs the full pipeline:
    Audio -> Transcription -> Task Extraction -> Task Assignment -> Output
    """
    # PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        description="Meeting Task Assignment System - Extract and assign tasks from meeting audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --audio meeting.wav --team team.json --output tasks.json
    python main.py -a meeting.mp3 -t my_team.json -o output.json
    python main.py --audio meeting.m4a --team team.json  # Uses default output.json
        """
    )
    
    parser.add_argument(
        '-a', '--audio',
        type=str,
        required=True,
        help='Path to the audio file (wav, mp3, or m4a)'
    )
    
    parser.add_argument(
        '-t', '--team',
        type=str,
        required=True,
        help='Path to the team members JSON file'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='tasks_output.json',
        help='Path for the output JSON file (default: tasks_output.json)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    
    args = parser.parse_args()
    
    # VALIDATE INPUTS
    print("\n" + "="*60)
    print("MEETING TASK ASSIGNMENT SYSTEM")
    print("="*60)
    
    # Check audio file
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Check file extension
    valid_extensions = ['.wav', '.mp3', '.m4a']
    file_ext = os.path.splitext(args.audio)[1].lower()
    if file_ext not in valid_extensions:
        print(f"Error: Unsupported audio format: {file_ext}")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    print(f"Audio file: {args.audio}")
    print(f"Team file: {args.team}")
    print(f"Output file: {args.output}")
    print(f"Whisper model: {args.model}")
    print("="*60)
    
    # STEP 1: LOAD TEAM MEMBERS
    print("\n[STEP 1/4] Loading team members...")
    try:
        team_members = load_team_members(args.team)
        print(f"Loaded {len(team_members)} team members:")
        for member in team_members:
            print(f"  - {member.name} ({member.role})")
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"Error loading team file: {e}")
        sys.exit(1)
    
    # STEP 2: TRANSCRIBE AUDIO
    print("\n[STEP 2/4] Transcribing audio...")
    try:
        transcriber = Transcriber(model_name=args.model)
        transcript = transcriber.transcribe(args.audio)
        
        print("\n--- TRANSCRIPT ---")
        print(transcript)
        print("--- END TRANSCRIPT ---")
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        sys.exit(1)
    
    # STEP 3: EXTRACT TASKS
    print("\n[STEP 3/4] Extracting tasks...")
    try:
        extractor = TaskExtractor()
        tasks = extractor.extract_tasks(transcript, team_members)
        
        if not tasks:
            print("Warning: No tasks were identified in the meeting.")
            print("This could mean:")
            print("  - The audio quality was poor")
            print("  - No task-related language was detected")
            print("  - The meeting didn't discuss specific tasks")
    except Exception as e:
        print(f"Error extracting tasks: {e}")
        sys.exit(1)
    
    # STEP 4: ASSIGN TASKS
    print("\n[STEP 4/4] Assigning tasks to team members...")
    try:
        assigner = TaskAssigner(team_members)
        assigned_tasks = assigner.assign_tasks(tasks)
    except Exception as e:
        print(f"Error assigning tasks: {e}")
        sys.exit(1)
    
    # OUTPUT RESULTS
    # Print to console
    print_tasks_table(assigned_tasks)
    
    # Save to JSON
    save_tasks_to_json(assigned_tasks, args.output)
    
    print("\nDone! Meeting tasks have been extracted and assigned.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

