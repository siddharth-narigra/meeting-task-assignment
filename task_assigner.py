"""
task_assigner.py - Custom Task Assignment Logic

This module assigns tasks to team members using custom logic.
No external AI APIs are used - everything is rule-based matching.

Assignment Strategy (in order of priority):
1. Direct Mention - If a name was mentioned with the task, assign to them
2. Skill Matching - Score each person by skill keyword overlap
3. Role Matching - Map task types to relevant roles

Why this approach?
- Direct mentions are explicit and should be honored
- Skill matching uses keyword overlap (simple but effective)
- Role matching provides fallback based on job titles
"""

import re
from typing import List, Dict, Optional, Tuple
from rapidfuzz import fuzz, process
from models import Task, TeamMember


def match_name_to_team(recognized_name: str, team_members: List[TeamMember], threshold: int = 60) -> Optional[str]:
    """
    Match a recognized name from transcript to a team member using fuzzy matching.
    
    Uses rapidfuzz for fast, production-grade fuzzy string matching.
    
    Args:
        recognized_name: Name as transcribed (may be misspelled)
        team_members: List of team members to match against
        threshold: Minimum match score (0-100), default 60%
        
    Returns:
        Matched team member name, or None if no good match
        
    Example:
        match_name_to_team("Rhea", team) → "Riya" (if Riya is in team)
        match_name_to_team("Roghav", team) → "Raghav"
    """
    if not recognized_name or not team_members:
        return None
    
    # Normalize the input
    recognized_name = recognized_name.strip().lower()
    
    # Build list of team member names
    team_names = [m.name for m in team_members]
    team_names_lower = [n.lower() for n in team_names]
    
    # Try exact match first (fast path)
    if recognized_name in team_names_lower:
        idx = team_names_lower.index(recognized_name)
        return team_names[idx]
    
    # Use rapidfuzz for fuzzy matching
    result = process.extractOne(
        recognized_name, 
        team_names_lower, 
        scorer=fuzz.ratio
    )
    
    if result and result[1] >= threshold:
        # result is (matched_string, score, index)
        idx = result[2]
        return team_names[idx]
    
    return None


def handle_someone_assignment(task: 'Task', team_members: List[TeamMember]) -> Optional[str]:
    """
    Handle implicit "someone" assignments by matching to role/skills.
    
    When transcript says "someone" or "let someone", try to find
    the best team member based on role mentions in the task.
    
    Args:
        task: The task with implicit assignment
        team_members: List of available team members
        
    Returns:
        Name of best matching team member, or None if no match
    """
    description_lower = task.description.lower()
    
    # Check if any role is mentioned: "someone from backend", "backend team"
    role_keywords = {
        'backend': ['backend', 'server', 'api', 'database'],
        'frontend': ['frontend', 'ui', 'interface', 'react'],
        'qa': ['qa', 'test', 'testing', 'quality'],
        'design': ['design', 'ux', 'figma', 'mockup'],
        'content': ['content', 'writing', 'documentation', 'faq', 'docs'],
    }
    
    for role_type, keywords in role_keywords.items():
        for keyword in keywords:
            if keyword in description_lower:
                # Find team member with matching role
                for member in team_members:
                    member_role_lower = member.role.lower()
                    if role_type in member_role_lower or any(kw in member_role_lower for kw in keywords):
                        return member.name
    
    # No role match found
    return None


class TaskAssigner:
    """
    Assigns tasks to team members using custom matching logic.
    
    The assignment algorithm:
    1. If already assigned (direct mention), keep it
    2. Otherwise, score each team member based on:
       - Skill keyword matches (e.g., "database" matches "Database" skill)
       - Role keyword matches (e.g., "bug" matches "Developer" role)
    3. Assign to the highest-scoring member
    """
    
    # ROLE KEYWORDS MAPPING
    # Maps task keywords to relevant roles
    # If a task contains these keywords, prioritize people with matching roles
    ROLE_KEYWORDS = {
        'frontend developer': [
            'frontend', 'front-end', 'ui bug', 'css', 'html', 'react', 
            'vue', 'angular', 'javascript', 'js', 'login', 'button',
            'form', 'page', 'screen', 'display', 'user interface'
        ],
        'backend developer': [
            'backend', 'back-end', 'api', 'server', 'database', 'db',
            'performance', 'optimization', 'cache', 'query', 'endpoint',
            'microservice', 'rest', 'graphql'
        ],
        'backend engineer': [
            'backend', 'back-end', 'api', 'server', 'database', 'db',
            'performance', 'optimization', 'cache', 'query', 'endpoint',
            'microservice', 'rest', 'graphql'
        ],
        'ui/ux designer': [
            'design', 'ui', 'ux', 'user experience', 'mockup', 'prototype',
            'figma', 'sketch', 'wireframe', 'onboarding', 'flow', 'layout',
            'visual', 'interface design', 'screens'
        ],
        'qa engineer': [
            'test', 'testing', 'qa', 'quality', 'unit test', 'automation',
            'bug', 'verify', 'validation', 'regression', 'coverage'
        ],
        'devops': [
            'deploy', 'deployment', 'ci/cd', 'pipeline', 'docker', 
            'kubernetes', 'infrastructure', 'monitoring', 'aws', 'cloud'
        ],
        'product manager': [
            'requirement', 'spec', 'documentation', 'roadmap', 'priority',
            'stakeholder', 'user story', 'acceptance criteria'
        ],
    }
    
    # SKILL BOOSTERS
    # These keywords in a task give extra weight to matching skills
    SKILL_BOOST_KEYWORDS = [
        'expert', 'good at', 'experienced', 'knows', 'worked on',
        'specialist', 'familiar with'
    ]
    
    def __init__(self, team_members: List[TeamMember]):
        """
        Initialize the assigner with the team.
        
        Args:
            team_members: List of available team members
        """
        self.team_members = team_members
        self.team_by_name = {m.name.lower(): m for m in team_members}
        
        print(f"TaskAssigner initialized with {len(team_members)} team members:")
        for member in team_members:
            print(f"  - {member.name} ({member.role}): {', '.join(member.skills)}")
    
    def calculate_skill_score(self, task: Task, member: TeamMember) -> Tuple[int, List[str]]:
        """
        Calculate how well a team member's skills match a task.
        
        We look for keyword overlaps between:
        - The task description/title
        - The member's skill list
        
        Args:
            task: The task to score
            member: The team member to evaluate
            
        Returns:
            Tuple of (score, list of matching skills)
        """
        task_text = f"{task.task} {task.description}".lower()
        score = 0
        matching_skills = []
        
        for skill in member.skills:
            skill_lower = skill.lower()
            
            # Check if skill keyword appears in task
            # We check each word in the skill (e.g., "UI bugs" -> check "ui" and "bugs")
            skill_words = skill_lower.split()
            
            for word in skill_words:
                if len(word) >= 3 and word in task_text:  # Skip very short words
                    score += 1
                    if skill not in matching_skills:
                        matching_skills.append(skill)
                    break  # Count each skill only once
        
        return score, matching_skills
    
    def calculate_role_score(self, task: Task, member: TeamMember) -> Tuple[int, str]:
        """
        Calculate how well a team member's role matches a task.
        
        We look up the member's role in ROLE_KEYWORDS and check
        if any of those keywords appear in the task.
        
        Args:
            task: The task to score
            member: The team member to evaluate
            
        Returns:
            Tuple of (score, reason string)
        """
        task_text = f"{task.task} {task.description}".lower()
        role_lower = member.role.lower()
        
        # Find keywords for this role
        role_keywords = []
        for role, keywords in self.ROLE_KEYWORDS.items():
            if role in role_lower or role_lower in role:
                role_keywords = keywords
                break
        
        if not role_keywords:
            return 0, ""
        
        # Count matches
        matches = []
        for keyword in role_keywords:
            if keyword in task_text:
                matches.append(keyword)
        
        if matches:
            reason = f"{member.role} matches task type"
            return len(matches), reason
        
        return 0, ""
    
    def find_best_assignee(self, task: Task) -> Tuple[Optional[TeamMember], str]:
        """
        Find the best team member to assign a task to.
        
        Scoring algorithm:
        1. Skill match score (how many skills overlap with task)
        2. Role match score (does their role fit the task type)
        3. Combined score determines the winner
        
        Args:
            task: The task to assign
            
        Returns:
            Tuple of (best member, reason for assignment)
        """
        if not self.team_members:
            return None, "No team members available"
        
        scores = []
        
        for member in self.team_members:
            # Calculate skill score
            skill_score, matching_skills = self.calculate_skill_score(task, member)
            
            # Calculate role score
            role_score, role_reason = self.calculate_role_score(task, member)
            
            # Combined score (skill matches are weighted higher)
            total_score = (skill_score * 2) + role_score
            
            # Build reason
            reasons = []
            if matching_skills:
                reasons.append(f"Skills: {', '.join(matching_skills)}")
            if role_reason:
                reasons.append(role_reason)
            
            reason = "; ".join(reasons) if reasons else "Default assignment"
            
            scores.append({
                'member': member,
                'score': total_score,
                'skill_score': skill_score,
                'role_score': role_score,
                'reason': reason
            })
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the best match
        best = scores[0]
        
        # If no one has any score, just pick the first available
        if best['score'] == 0:
            return best['member'], "Assigned as available team member"
        
        return best['member'], best['reason']
    
    def resolve_dependencies(self, tasks: List[Task]) -> None:
        """
        Try to link tasks that depend on other tasks.
        
        We look for patterns like "depends on login bug" and try to
        find a matching task about "login bug".
        
        This modifies tasks in place, adding dependency IDs.
        
        Args:
            tasks: List of all tasks (modified in place)
        """
        for task in tasks:
            # Check if this task has a dependency flag
            if not getattr(task, '_has_dependency', False):
                continue
            
            description_lower = task.description.lower()
            
            # Try to find what it depends on
            for other_task in tasks:
                if other_task.id == task.id:
                    continue
                
                # Check if the other task is mentioned in the dependency
                other_title_words = other_task.task.lower().split()
                
                # Look for significant words from other task in this task's description
                matches = 0
                for word in other_title_words:
                    if len(word) >= 4 and word in description_lower:  # Skip short words
                        matches += 1
                
                # If enough words match, consider it a dependency
                if matches >= 2:
                    # Avoid duplicate dependencies
                    if other_task.id not in task.dependencies:
                        task.dependencies.append(other_task.id)
                    break  # Usually only one dependency mentioned
    
    def assign_tasks(self, tasks: List[Task]) -> List[Task]:
        """
        Main method: Assign all tasks to team members.
        
        This is the entry point that:
        1. Keeps existing assignments (from direct mentions)
        2. Assigns unassigned tasks using skill/role matching
        3. Resolves dependencies between tasks
        
        Args:
            tasks: List of tasks (may have some already assigned)
            
        Returns:
            The same list with all tasks assigned
        """
        print("\nAssigning tasks to team members...")
        
        for task in tasks:
            # If already assigned (direct mention), validate against team
            if task.assigned_to:
                # Verify assignment exists in team (handles fuzzy matching edge cases)
                matched = match_name_to_team(task.assigned_to, self.team_members)
                if matched:
                    task.assigned_to = matched  # Use canonical name
                    print(f"  Task #{task.id}: Already assigned to {task.assigned_to}")
                    if not task.reason:
                        task.reason = "Directly mentioned in meeting"
                    continue
                else:
                    # Name doesn't match any team member, clear and reassign
                    print(f"  Task #{task.id}: '{task.assigned_to}' not in team, will reassign")
                    task.assigned_to = None
            
            # Check for "someone" implicit assignment
            if 'someone' in task.description.lower():
                someone_match = handle_someone_assignment(task, self.team_members)
                if someone_match:
                    task.assigned_to = someone_match
                    task.reason = "Matched by role (implicit 'someone' assignment)"
                    task.flag_for_review = True
                    print(f"  Task #{task.id}: Assigned to {someone_match} (someone → role match)")
                    continue
            
            # Find the best person for this task based on skills/role
            best_member, reason = self.find_best_assignee(task)
            
            if best_member:
                task.assigned_to = best_member.name
                task.reason = reason
                print(f"  Task #{task.id}: Assigned to {best_member.name} ({reason})")
            else:
                print(f"  Task #{task.id}: Could not assign (no team members)")
        
        # Resolve dependencies
        print("\nResolving task dependencies...")
        self.resolve_dependencies(tasks)
        
        for task in tasks:
            if task.dependencies:
                print(f"  Task #{task.id} depends on Task(s) #{task.dependencies}")
        
        print("\nTask assignment complete!")
        
        return tasks


# This allows testing the module directly
if __name__ == "__main__":
    from models import Task, TeamMember
    
    # Sample team members (from task.md)
    team = [
        TeamMember("Sakshi", "Frontend Developer", ["React", "JavaScript", "UI bugs"]),
        TeamMember("Mohit", "Backend Engineer", ["Database", "APIs", "Performance optimization"]),
        TeamMember("Arjun", "UI/UX Designer", ["Figma", "User flows", "Mobile design"]),
        TeamMember("Lata", "QA Engineer", ["Testing", "Automation", "Quality assurance"]),
    ]
    
    # Sample tasks (simulating output from task_extractor)
    tasks = [
        Task(
            id=1,
            task="Fix critical login bug",
            description="We need someone to fix the critical login bug that users reported",
            assigned_to="Sakshi",  # Already assigned (mentioned directly)
            deadline="Tomorrow evening",
            priority="Critical",
            reason="Directly mentioned"
        ),
        Task(
            id=2,
            task="Optimize database performance",
            description="The database performance is really slow, needs backend optimization",
            deadline="End of this week",
            priority="High"
        ),
        Task(
            id=3,
            task="Update API documentation",
            description="We need to update the API documentation before Friday's release",
            deadline="Friday",
            priority="High"
        ),
        Task(
            id=4,
            task="Design new onboarding screens",
            description="Someone should design the new onboarding screens for the next sprint",
            deadline="Next Monday",
            priority="Medium"
        ),
        Task(
            id=5,
            task="Write unit tests for payment module",
            description="We need to write unit tests for the payment module, depends on login bug fix",
            deadline="Wednesday",
            priority="Medium"
        ),
    ]
    
    # Mark task 5 as having a dependency
    tasks[4]._has_dependency = True
    
    # Test assignment
    assigner = TaskAssigner(team)
    assigned_tasks = assigner.assign_tasks(tasks)
    
    print("\n" + "="*60)
    print("ASSIGNED TASKS:")
    print("="*60)
    for task in assigned_tasks:
        print(f"\n#{task.id}: {task.task}")
        print(f"   Assigned to: {task.assigned_to}")
        print(f"   Reason: {task.reason}")
        print(f"   Dependencies: {task.dependencies or 'None'}")

