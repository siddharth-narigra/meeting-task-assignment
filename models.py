"""
models.py - Data structures for the Meeting Task Assignment System

This module defines the core data classes that represent:
1. TeamMember - A person who can be assigned tasks
2. Task - A task extracted from the meeting transcript

We use Python dataclasses because they:
- Automatically generate __init__, __repr__, __eq__ methods
- Make the code cleaner and more readable
- Provide type hints for better IDE support
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TeamMember:
    """
    Represents a team member who can be assigned tasks.
    
    Attributes:
        name: The person's name (e.g., "Sakshi")
        role: Their job title (e.g., "Frontend Developer")
        skills: List of skills they possess (e.g., ["React", "JavaScript", "UI bugs"])
    
    Example:
        member = TeamMember(
            name="Sakshi",
            role="Frontend Developer", 
            skills=["React", "JavaScript", "UI bugs"]
        )
    """
    name: str
    role: str
    skills: List[str] = field(default_factory=list)
    
    def has_skill(self, keyword: str) -> bool:
        """
        Check if this team member has a skill matching the keyword.
        
        Uses case-insensitive partial matching.
        For example, if skills = ["React", "JavaScript"], 
        has_skill("react") returns True
        has_skill("java") returns True (partial match with JavaScript)
        
        Args:
            keyword: The skill keyword to search for
            
        Returns:
            True if any skill contains the keyword, False otherwise
        """
        keyword_lower = keyword.lower()
        for skill in self.skills:
            if keyword_lower in skill.lower():
                return True
        return False


@dataclass
class Task:
    """
    Represents a task extracted from the meeting transcript.
    
    Attributes:
        id: Unique identifier for the task (1, 2, 3, etc.)
        task: Short title of the task (e.g., "Fix critical login bug")
        description: The original sentence(s) from the transcript
        assigned_to: Name of the team member assigned (can be None if unassigned)
        deadline: When the task is due (e.g., "Tomorrow evening", "End of this week")
        priority: Priority level - "Critical", "High", "Medium", or "Low"
        dependencies: List of task IDs this task depends on
        reason: Why this person was assigned (for transparency)
    
    Example:
        task = Task(
            id=1,
            task="Fix critical login bug",
            description="we need someone to fix the critical login bug...",
            assigned_to="Sakshi",
            deadline="Tomorrow evening",
            priority="Critical",
            dependencies=[],
            reason="Frontend task, directly mentioned"
        )
    """
    id: int
    task: str
    description: str
    assigned_to: Optional[str] = None
    deadline: Optional[str] = None
    priority: str = "Medium"
    dependencies: List[int] = field(default_factory=list)
    reason: Optional[str] = None
    # Internal field for tracking if task has unresolved dependency mention
    _has_dependency: bool = field(default=False, repr=False)
    
    def to_dict(self) -> dict:
        """
        Convert the Task to a dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the task
        """
        return {
            "id": self.id,
            "task": self.task,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "deadline": self.deadline,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "reason": self.reason
        }

