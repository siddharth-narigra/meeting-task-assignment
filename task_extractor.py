"""
task_extractor.py - Custom Task Identification Logic

This module contains the CUSTOM NLP logic to extract tasks from meeting transcripts.
No external AI APIs are used here - everything is rule-based pattern matching.

The extraction pipeline:
1. Sentence Segmentation - Break transcript into sentences using spaCy
2. Task Detection - Find sentences that describe tasks using keyword patterns
3. Context Merging - Merge follow-up sentences (context) with their parent task
4. Deadline Extraction - Find time expressions like "tomorrow", "by Friday"
5. Priority Detection - Find urgency indicators like "critical", "urgent"
6. Dependency Detection - Find phrases like "depends on", "after"
7. Person Mention - Detect team member names using FUZZY matching

Why pattern-based approach?
- Meeting language follows predictable patterns
- Task assignments use common phrases ("need to", "should", "has to")
- Deadlines use common time expressions ("tomorrow", "next week")
- No API costs or external dependencies for the core logic

IMPROVEMENTS v2:
- Fuzzy name matching to handle Whisper transcription errors (Sakshi → Sokshi)
- Context sentence merging (combines "fix bug" + "this needs to be done by tomorrow")
- False positive filtering (excludes intro sentences and questions)

IMPROVEMENTS v3:
- Transcript normalization (fixes "Friday'S" → "Friday's", weird apostrophes, etc.)
- Priority scoring system (considers deadline urgency + impact keywords, not just "critical")
"""

import re
import spacy  # type: ignore
import dateparser
from datetime import datetime
from typing import List, Tuple, Optional, Any
from rapidfuzz import fuzz, process
from models import Task, TeamMember


# =============================================================================
# TRANSCRIPT NORMALIZATION
# =============================================================================
def normalize_transcript(transcript: str) -> str:
    """
    Clean and normalize the transcript text before processing.
    
    This fixes common issues from Whisper transcription:
    1. Weird apostrophes: ' ` → '
    2. Broken contractions: "it' s" → "it's", "Friday'S" → "Friday's"
    3. Extra whitespace
    4. Inconsistent quotes
    
    Args:
        transcript: Raw transcript from Whisper
        
    Returns:
        Cleaned transcript text
        
    Examples:
        "Friday'S release" → "Friday's release"
        "it' s blocking" → "it's blocking"
        "don` t forget" → "don't forget"
    """
    t = transcript
    
    # Step 1: Normalize different apostrophe characters to standard '
    # Whisper sometimes uses curly quotes or backticks
    t = t.replace("'", "'")  # Right single quote → apostrophe
    t = t.replace("'", "'")  # Left single quote → apostrophe  
    t = t.replace("`", "'")  # Backtick → apostrophe
    t = t.replace("´", "'")  # Acute accent → apostrophe
    
    # Step 2: Fix possessives with wrong case: "Friday'S" → "Friday's"
    # This pattern finds apostrophe followed by uppercase S and fixes it
    t = re.sub(r"'S\b", "'s", t)
    
    # Step 3: Fix broken contractions with space: "it' s" → "it's"
    t = re.sub(r"(\w)'\s+s\b", r"\1's", t, flags=re.IGNORECASE)
    t = re.sub(r"(\w)'\s+t\b", r"\1't", t, flags=re.IGNORECASE)  # "don' t" → "don't"
    t = re.sub(r"(\w)'\s+re\b", r"\1're", t, flags=re.IGNORECASE)  # "we' re" → "we're"
    t = re.sub(r"(\w)'\s+ll\b", r"\1'll", t, flags=re.IGNORECASE)  # "we' ll" → "we'll"
    t = re.sub(r"(\w)'\s+ve\b", r"\1've", t, flags=re.IGNORECASE)  # "I' ve" → "I've"
    
    # Step 4: Normalize quotes
    t = t.replace(""", '"').replace(""", '"')  # Curly quotes → straight
    t = t.replace("„", '"')  # German quote → straight
    
    # Step 5: Collapse multiple spaces into single space
    t = re.sub(r"\s+", " ", t)
    
    # Step 6: Clean up spaces around punctuation
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)  # Remove space before punctuation
    
    # Step 7: Strip leading/trailing whitespace
    t = t.strip()
    
    return t


def fuzzy_match(name: str, text: str, threshold: float = 0.7) -> bool:
    """
    Check if a name appears in text using fuzzy matching.
    
    This handles Whisper transcription errors like:
    - "Sakshi" → "Sokshi" (one character difference)
    - "Arjun" → "Our June" (sounds similar)
    
    Uses a simple approach:
    1. First try exact match (fast path)
    2. Then try phonetic/substring matching for common errors
    
    Args:
        name: The correct name to search for
        text: The text that might contain a misspelled version
        threshold: Minimum similarity ratio (0.0 to 1.0)
        
    Returns:
        True if name (or similar) found in text
    """
    name_lower = name.lower()
    text_lower = text.lower()
    
    # Fast path: exact match
    if name_lower in text_lower:
        return True
    
    # Check each word in the text for similarity
    words = re.findall(r'\b\w+\b', text_lower)
    
    for word in words:
        # Skip very short words
        if len(word) < 3:
            continue
            
        # Calculate similarity ratio
        similarity = calculate_similarity(name_lower, word)
        if similarity >= threshold:
            return True
        
        # Also check if name starts similarly (first 3+ chars match)
        # This catches "Sakshi" vs "Sokshi" where first letter differs
        if len(name_lower) >= 3 and len(word) >= 3:
            # Check if most characters match
            matches = sum(1 for a, b in zip(name_lower, word) if a == b)
            if matches >= len(name_lower) * 0.6:  # 60% of chars match
                return True
    
    # Special handling for multi-word transcription errors
    # "Arjun" might become "Our June" (two words)
    # Check consecutive word pairs
    for i in range(len(words) - 1):
        combined = words[i] + words[i + 1]
        # Remove spaces and check similarity
        similarity = calculate_similarity(name_lower, combined)
        if similarity >= threshold:
            return True
    
    return False


def calculate_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between two strings using Levenshtein-like approach.
    
    Returns a ratio from 0.0 (completely different) to 1.0 (identical).
    
    This is a simplified version that:
    - Counts matching characters
    - Penalizes length differences
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    if s1 == s2:
        return 1.0
    
    if not s1 or not s2:
        return 0.0
    
    # Count character matches (order-independent for robustness)
    # This handles transpositions and substitutions
    len1, len2 = len(s1), len(s2)
    
    # Use the shorter string as reference
    shorter, longer = (s1, s2) if len1 <= len2 else (s2, s1)
    
    # Count how many characters from shorter appear in longer
    matches = 0
    longer_chars = list(longer)
    for char in shorter:
        if char in longer_chars:
            matches += 1
            longer_chars.remove(char)  # Don't count same char twice
    
    # Calculate similarity as ratio of matches to average length
    avg_len = (len1 + len2) / 2
    similarity = matches / avg_len
    
    # Penalize large length differences
    len_diff_penalty = abs(len1 - len2) / max(len1, len2)
    similarity = similarity * (1 - len_diff_penalty * 0.5)
    
    return min(similarity, 1.0)


class TaskExtractor:
    """
    Extracts tasks from meeting transcripts using custom NLP rules.
    
    This class implements pattern-based extraction without external AI APIs.
    It uses spaCy for sentence segmentation and regex for pattern matching.
    
    v2 Improvements:
    - Fuzzy name matching for Whisper transcription errors
    - Context sentence merging
    - False positive filtering
    """
    
    # ==========================================================================
    # TASK DETECTION PATTERNS
    # ==========================================================================
    # These phrases typically indicate a task is being assigned
    TASK_PATTERNS = [
        r'\bneed(?:s)?\s+to\b',           # "need to", "needs to"
        r'\bneeds?\s+someone\s+to\b',     # "need someone to", "needs someone to"  
        r'\bshould\b',                     # "should"
        r'\bhas\s+to\b',                   # "has to"
        r'\bhave\s+to\b',                  # "have to"
        r'\bmust\b',                       # "must"
        r'\bwe\s+need\b',                  # "we need"
        r'\bwe\s+should\b',                # "we should"
        r'\bgoing\s+to\b',                 # "going to"
        r'\bplan\s+to\b',                  # "plan to"
        r'\btackle\b',                     # "tackle"
        r'\bwork\s+on\b',                  # "work on"
        r'\bhandle\b',                     # "handle"
        # MVP additions
        r'\binvestigate\b',                # "investigate the issue"
        r'\bprepare\s+(?:a\s+)?report\b',  # "prepare a report"
        r'\bpick\s+(?:that\s+)?up\b',      # "pick that up", "pick up"
        r'\bfinish\s+(?:it|this|that)?\b', # "finish it", "finish this"
        r'\bdraft\b',                       # "draft the FAQ"
        r'\bcan\s+you\s+(?:handle|do|take|finish)\b',  # "can you handle that"
        r'\blet\s+\w+\s+(?:handle|do|take)\b',  # "let Karan handle"
        r'\bwill\s+pick\s+(?:that\s+)?up\b',  # "will pick that up"
        r'\bstart\s+(?:on|with)?\b',       # "start on the testing"
        # Direct addressing patterns: "Name, verb..." 
        r'^[A-Z][a-z]+,\s+\w+',            # "Mehul, update..." - name at start
        r'\bupdate\s+(?:all\s+)?(?:the\s+)?\w+',  # "update all environment keys"
        r'\bregenerate\b',                  # "regenerate the screenshots"
        r'\bpush\s+(?:a\s+)?(?:fixed\s+)?\w+',  # "push a fixed build"
        r'\bsend\s+(?:the\s+)?(?:\w+\s+)*to\b',  # "send the approved version to"
        r'\btake\s+that\b',                 # "take that and..."
        r'\breview\s+(?:the\s+)?\w+',       # "review the legal copy"
    ]
    
    # Action verbs that indicate a task when combined with an object
    # These are checked more carefully to avoid false positives
    ACTION_VERB_PATTERNS = [
        r'\bfix\s+(?:the\s+)?(?:\w+\s+){0,2}(?:bug|issue|problem|error)\b',  # "fix the login bug"
        r'\bupdate\s+(?:the\s+)?(?:\w+\s+){0,3}(?:doc|documentation|docs|faq)\b',  # "update the API documentation"
        r'\bwrite\s+(?:\w+\s+){0,2}(?:test|tests|unit test)\b',  # "write unit tests"
        r'\bdesign\s+(?:the\s+)?(?:new\s+)?(?:\w+\s+){0,2}(?:screen|ui|interface|page|flow)\b',  # "design the new screens"
        r'\boptimize\s+(?:the\s+)?(?:\w+\s+){0,2}(?:database|performance|query|api)\b',  # "optimize database"
        r'\bcreate\s+(?:the\s+)?(?:new\s+)?\w+\b',  # "create the new X"
        r'\bimplement\s+(?:the\s+)?(?:new\s+)?\w+\b',  # "implement the feature"
        # MVP additions
        r'\binvestigate\s+(?:the\s+)?(?:\w+\s+){0,3}(?:root\s+cause|issue|problem|bug|error|logs?|spikes?)\b',
        r'\bprepare\s+(?:a\s+)?(?:\w+\s+){0,2}(?:report|summary|document)\b',
        r'\bdraft\s+(?:the\s+)?(?:\w+\s+){0,2}(?:faq|updates?|docs?|documentation)\b',
        r'\bredesign\s+(?:the\s+)?(?:\w+\s+){0,2}(?:homepage|page|screen|ui)\b',
        r'\bfinalize\s+(?:the\s+)?(?:\w+\s+){0,2}\w+\b',  # "finalize the homepage"
        r'\btest(?:ing)?\s+(?:the\s+)?(?:for\s+)?(?:\w+\s+){0,2}(?:module|feature|cart|checkout)\b',
    ]
    
    # ==========================================================================
    # FALSE POSITIVE PATTERNS (sentences to EXCLUDE)
    # ==========================================================================
    # These sentences should NOT be treated as tasks
    # NOTE: We do NOT blanket-filter all questions (r'\?$') because some questions
    # contain direct assignments like "Ananya, can you handle that by tomorrow?"
    FALSE_POSITIVE_PATTERNS = [
        r'^(?:hi|hello|hey)\s+everyone',           # Greetings: "Hi everyone"
        r'^let\'?s\s+discuss',                     # "Let's discuss" (intro, not task)
        r'^let\'?s\s+talk\s+about',               # "Let's talk about"
        r'\bdidn\'?t\s+you\s+work\b',             # Questions: "didn't you work on..."
        r'\byou\'?re\s+good\s+(?:at|with)\b',     # "you're good with X" (not a task)
        r'^what\s+(?:do|does|is|are)\b.*\?$',     # "What do you think?" type questions
        r'^how\s+(?:do|does|is|are|about)\b.*\?$', # "How is the progress?" type questions
        r'^why\s+(?:do|does|is|are)\b.*\?$',       # "Why is this broken?" type questions
    ]
    
    # ==========================================================================
    # ASSIGNMENT QUESTION PATTERNS
    # ==========================================================================
    # Questions that ARE task assignments - these should be treated as tasks
    # even though they end with "?"
    ASSIGNMENT_QUESTION_PATTERNS = [
        r'\bcan\s+you\s+(?:handle|do|take|finish|work\s+on)\b.*\?$',  # "can you handle that?"
        r'\bwill\s+you\s+(?:handle|do|take|finish|work\s+on)\b.*\?$', # "will you handle that?"
        r'\bcould\s+you\s+(?:handle|do|take|finish|work\s+on)\b.*\?$', # "could you handle that?"
    ]
    
    # ==========================================================================
    # CONTEXT SENTENCE PATTERNS
    # ==========================================================================
    # Sentences that provide context for the PREVIOUS task (should be merged)
    # These start with pronouns referring to something mentioned before
    CONTEXT_PATTERNS = [
        r'^this\s+(?:needs?|should|has|is|depends?|can)\b',  # "This needs to be done..."
        r'^it\s+(?:needs?|should|has|is)\b',                  # "It should be done by..."
        r'^it\'?s\s+(?:blocking|affecting|important)\b',     # "It's blocking users"
        r'^(?:so\s+)?let\'?s\s+plan\b',                       # "So let's plan this..."
        r'^this\s+depends\b',                                  # "This depends on..."
        r'^that\s+(?:needs?|should|can)\b',                   # "That needs to..."
        r'^don\'?t\s+start\s+until\b',                        # "Don't start until..."
        r'^wait\s+(?:for|until)\b',                           # "Wait for...", "Wait until..."
    ]
    
    # ==========================================================================
    # DEADLINE PATTERNS
    # ==========================================================================
    DEADLINE_PATTERNS = [
        # Today patterns
        (r'\bby\s+today\b', 'Today'),
        (r'\btoday\b', 'Today'),
        (r'\btonight\b', 'Tonight'),
        (r'\bby\s+end\s+of\s+today\b', 'End of today'),
        
        # Specific times
        (r'\bby\s+tomorrow\s+(?:morning|afternoon|evening|night)\b', None),
        (r'\btomorrow\s+(?:morning|afternoon|evening|night)\b', None),
        (r'\bby\s+tomorrow\b', 'Tomorrow'),
        (r'\btomorrow\b', 'Tomorrow'),
        
        # Day of week with time of day (must come before plain day patterns)
        (r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:morning|afternoon|evening|night)\b', None),
        (r'\bfor\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+(?:morning|afternoon|evening|night)\b', None),
        
        # "before next Thursday's announcement" pattern
        (r'\bbefore\s+next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\'?s)?\b', None),
        (r'\bbefore\s+(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\'?s)?\b', None),
        
        # Day of week
        (r'\bby\s+(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        (r'\bfor\s+(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        (r'\b(?:until|til)\s+(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        (r'\bwaits?\s+until\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        (r'\bcan\s+wait\s+until\s+(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        
        # Relative time
        (r'\bend\s+of\s+(?:this\s+)?week\b', 'End of this week'),
        (r'\bend\s+of\s+(?:the\s+)?day\b', 'End of day'),
        (r'\bnext\s+week\b', 'Next week'),
        (r'\bnext\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        (r'\bthis\s+week\b', 'This week'),
        (r'\bnext\s+sprint\b', 'Next sprint'),
        (r'\bnext\s+month\b', 'Next month'),
        
        # Standalone weekday (for "so QA will pick that up Wednesday")
        (r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', None),
        
        # ASAP / Urgent timing
        (r'\basap\b', 'ASAP'),
        (r'\bas\s+soon\s+as\s+possible\b', 'ASAP'),
        (r'\bimmediately\b', 'Immediately'),
        (r'\bright\s+away\b', 'Right away'),
        (r'\burgently\b', 'Urgently'),
    ]
    
    # ==========================================================================
    # PRIORITY MAPPING (Simple keyword-based)
    # ==========================================================================
    # Maps keywords to priority levels - first match wins
    PRIORITY_MAP = {
        # Critical
        'critical': 'Critical',
        'urgent': 'Critical',
        'asap': 'Critical',
        'blocker': 'Critical',
        'blocking': 'Critical',
        'emergency': 'Critical',
        'showstopper': 'Critical',
        # High
        'prioritize': 'High',
        'prioritise': 'High',  # British spelling
        'important': 'High',
        'high priority': 'High',
        'before release': 'High',
        'by thursday': 'High',
        'by friday': 'High',
        'before thursday': 'High',
        'before friday': 'High',
        # Medium (default, but can be explicitly mentioned)
        'wednesday': 'Medium',
        'next monday': 'Medium',
        'end of week': 'Medium',
        'this week': 'Medium',
        # Low
        'no rush': 'Low',
        'can wait': 'Low',
        'low priority': 'Low',
        'when possible': 'Low',
        'nice to have': 'Low',
        'eventually': 'Low',
    }
    
    # ==========================================================================
    # DEPENDENCY PATTERNS
    # ==========================================================================
    DEPENDENCY_PATTERNS = [
        r'\bdepends?\s+on\b',              # "depends on"
        r'\bafter\s+(?:the\s+)?(?:\w+\s+){0,3}(?:is\s+)?(?:done|completed?|finished)\b',
        r'\bonce\s+(?:the\s+)?(?:\w+\s+){0,3}(?:is\s+)?(?:done|completed?|finished)\b',
        r'\bwaiting\s+(?:for|on)\b',       # "waiting for", "waiting on"
        r'\bblocked\s+by\b',               # "blocked by"
        r'\bfirst\s+(?:we\s+)?need\b',     # "first need", "first we need"
        # MVP additions
        r'\bcan\s+only\s+start\s+after\b', # "can only start after"
        r'\bonly\s+start\s+after\b',       # "only start after"
        r'\bafter\s+\w+\s+pushes?\b',      # "after Riya pushes"
        r'\bafter\s+\w+\s+finishes?\b',    # "after Karan finishes"
        r'\buntil\s+\w+(?:\'s)?\s+\w+\s+(?:passes?|completes?|finishes?)\b',  # "until Mehul's build passes"
    ]
    
    def __init__(self):
        """Initialize the TaskExtractor with spaCy language model."""
        print("Loading spaCy language model...")
        self.nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded!")
        
        # Compile regex patterns for efficiency
        self.compiled_task_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.TASK_PATTERNS
        ]
        
        self.compiled_action_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.ACTION_VERB_PATTERNS
        ]
        
        self.compiled_false_positive_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.FALSE_POSITIVE_PATTERNS
        ]
        
        self.compiled_assignment_question_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.ASSIGNMENT_QUESTION_PATTERNS
        ]
        
        self.compiled_context_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CONTEXT_PATTERNS
        ]
        
        self.compiled_dependency_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.DEPENDENCY_PATTERNS
        ]
    
    def extract_sentences(self, transcript: str) -> List[str]:
        """Split transcript into sentences using spaCy."""
        doc = self.nlp(transcript)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences
    
    def is_assignment_question(self, sentence: str) -> bool:
        """
        Check if a sentence is an assignment question (task disguised as question).
        
        Examples:
        - "Ananya, can you handle that by tomorrow evening?"
        - "Will you finish the report by Friday?"
        
        These should be treated as tasks, not filtered out.
        """
        for pattern in self.compiled_assignment_question_patterns:
            if pattern.search(sentence):
                return True
        return False
    
    def is_false_positive(self, sentence: str) -> bool:
        """
        Check if a sentence is a false positive (should NOT be a task).
        
        Filters out:
        - Greetings ("Hi everyone")
        - Discussion intros ("Let's discuss")
        - Rhetorical questions ("didn't you work on...?")
        
        But KEEPS assignment questions like "can you handle that?"
        """
        # First check if it's an assignment question - these are NOT false positives
        if self.is_assignment_question(sentence):
            return False
        
        for pattern in self.compiled_false_positive_patterns:
            if pattern.search(sentence):
                return True
        return False
    
    def is_context_sentence(self, sentence: str) -> bool:
        """
        Check if a sentence is providing context for a previous task.
        
        Context sentences start with:
        - "This needs to be done by..."
        - "It's blocking users"
        - "So let's plan this for..."
        
        These should be MERGED with the previous task, not created as new tasks.
        """
        for pattern in self.compiled_context_patterns:
            if pattern.search(sentence):
                return True
        return False
    
    def is_task_sentence(self, sentence: str) -> bool:
        """
        Determine if a sentence describes a NEW task.
        
        Checks:
        1. Not a false positive (greeting, question, etc.)
        2. Contains task indicator patterns OR action verbs with objects
        """
        # First, filter out false positives
        if self.is_false_positive(sentence):
            return False
        
        # Check standard task patterns
        for pattern in self.compiled_task_patterns:
            if pattern.search(sentence):
                return True
        
        # Check action verb patterns (more specific)
        for pattern in self.compiled_action_patterns:
            if pattern.search(sentence):
                return True
        
        return False
    
    def extract_deadline(self, text: str) -> Optional[str]:
        """Extract deadline from text (can be from task or context sentence)."""
        text_lower = text.lower()
        
        for pattern, replacement in self.DEADLINE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                deadline = replacement if replacement else match.group(0)
                # Use smart_title instead of .title() to avoid "Friday'S" bug
                return self.smart_title(deadline)
        
        return None
    
    def parse_deadline_to_iso(self, deadline_text: str, base_date: Optional[datetime] = None) -> Optional[str]:
        """
        Parse a deadline text to ISO 8601 format using dateparser.
        
        Args:
            deadline_text: Natural language deadline like "tomorrow evening", "next Thursday"
            base_date: Reference date for relative parsing (defaults to now)
            
        Returns:
            ISO 8601 formatted date string, or None if parsing fails
            
        Examples:
            "tomorrow evening" → "2025-12-01T18:00:00+05:30"
            "next Thursday" → "2025-12-04T00:00:00+05:30"
        """
        if not deadline_text:
            return None
        
        try:
            dt = dateparser.parse(
                deadline_text,
                settings={
                    'RELATIVE_BASE': base_date or datetime.now(),
                    'TIMEZONE': 'Asia/Kolkata',
                    'RETURN_AS_TIMEZONE_AWARE': True,
                    'PREFER_DATES_FROM': 'future',
                }  # type: ignore
            )
            if dt:
                return dt.isoformat()
        except Exception:
            pass
        
        return None
    
    def smart_title(self, text: str) -> str:
        """
        Capitalize text properly without messing up apostrophes.
        
        Python's .title() capitalizes after every apostrophe:
        "friday's" → "Friday'S" ← BAD
        
        This function keeps lowercase after apostrophes:
        "friday's" → "Friday's" ← GOOD
        
        Args:
            text: Text to capitalize
            
        Returns:
            Properly capitalized text
        """
        words = text.split()
        result = []
        
        for word in words:
            if "'" in word:
                # Handle words with apostrophes: "friday's" → "Friday's"
                parts = word.split("'")
                # Capitalize first part, keep rest lowercase
                capitalized = parts[0].capitalize() + "'" + "'".join(parts[1:]).lower()
                result.append(capitalized)
            else:
                # Normal word: capitalize first letter
                result.append(word.capitalize())
        
        return " ".join(result)
    
    def extract_priority(self, text: str, deadline: Optional[str] = None, 
                         has_critical_dependency: bool = False) -> str:
        """
        Extract priority using simple keyword matching.
        
        Uses PRIORITY_MAP for keyword-based detection.
        First match wins - keywords are ordered by priority.
        
        Args:
            text: The task text to analyze
            deadline: Optional deadline string (used for urgency boost)
            has_critical_dependency: True if task depends on a Critical/High task
            
        Returns:
            Priority level: "Critical", "High", "Medium", or "Low"
        """
        text_lower = text.lower()
        
        # Also check the deadline text if provided
        if deadline:
            text_lower += " " + deadline.lower()
        
        # Check for keywords in priority order (Critical first, then High, etc.)
        # The PRIORITY_MAP is ordered so this works correctly
        for keyword, priority in self.PRIORITY_MAP.items():
            if keyword in text_lower:
                # If depending on Critical/High task, boost Medium to High
                if has_critical_dependency and priority == "Medium":
                    return "High"
                return priority
        
        # Boost for critical dependency
        if has_critical_dependency:
            return "High"
        
        # Default to Medium (not Low) - safer default
        return "Medium"
    
    def has_dependency(self, text: str) -> bool:
        """Check if text mentions a dependency."""
        for pattern in self.compiled_dependency_patterns:
            if pattern.search(text):
                return True
        return False
    
    def find_mentioned_person(self, sentence: str, team_members: List[TeamMember]) -> Optional[str]:
        """
        Find if any team member is mentioned in the sentence.
        
        Uses rapidfuzz for robust fuzzy matching to handle Whisper errors:
        - "Sokshi" matches "Sakshi"
        - "Rhea" matches "Riya"
        - "Roghav" matches "Raghav"
        """
        if not team_members:
            return None
        
        # Build list of team member names
        team_names = [m.name for m in team_members]
        
        # Extract words from sentence that could be names (capitalized or at start)
        words = re.findall(r'\b[A-Za-z]+\b', sentence)
        
        for word in words:
            # Skip very short words or common words
            if len(word) < 3:
                continue
            
            # Try exact match first (fast path)
            word_lower = word.lower()
            for name in team_names:
                if word_lower == name.lower():
                    return name
            
            # Try fuzzy match with rapidfuzz
            result = process.extractOne(
                word_lower,
                [n.lower() for n in team_names],
                scorer=fuzz.ratio
            )
            
            if result and result[1] >= 65:  # 65% threshold
                idx = result[2]
                return team_names[idx]
        
        return None
    
    def correct_name_in_text(self, text: str, correct_name: str) -> str:
        """
        Replace misspelled name with correct name in text.
        
        IMPORTANT: Only replaces names at SPECIFIC POSITIONS where names
        typically appear in meeting transcripts:
        1. First word of the text
        2. Word immediately after a comma
        3. Word after greeting words (hey, hi, oh, also)
        
        This prevents replacing random words like "with" or "this" that
        might have superficial similarity to names.
        
        Args:
            text: The text containing a misspelled name
            correct_name: The correct spelling of the name
            
        Returns:
            Text with the name corrected at valid positions only
            
        Example:
            "Sokshi, we need you to fix this with tools" 
            → "Sakshi, we need you to fix this with tools"
            (only "Sokshi" replaced, not "this" or "with")
        """
        if not correct_name:
            return text
        
        name_lower = correct_name.lower()
        name_len = len(correct_name)
        
        # Split into words while tracking positions
        words = text.split()
        if not words:
            return text
        
        corrected_words = []
        
        # Greeting/transition words after which a name might appear
        name_trigger_words = {'hey', 'hi', 'hello', 'oh', 'also', 'and', 'ok', 'okay', 'so'}
        
        for i, word in enumerate(words):
            should_check = False
            
            # Position 1: First word of text
            if i == 0:
                should_check = True
            
            # Position 2: Word after a comma (previous word ends with comma)
            elif i > 0 and corrected_words[i-1].rstrip().endswith(','):
                should_check = True
            
            # Position 3: Word after greeting/transition words
            elif i > 0:
                prev_word_clean = re.sub(r'[^\w]', '', corrected_words[i-1].lower())
                if prev_word_clean in name_trigger_words:
                    should_check = True
            
            if should_check:
                # Check if this word is a misspelled version of the name
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                # Length check: word should be similar length to name (±2 chars)
                if len(clean_word) >= 3 and abs(len(clean_word) - name_len) <= 2:
                    similarity = calculate_similarity(name_lower, clean_word)
                    
                    # Use higher threshold (75%) for more accuracy
                    if similarity >= 0.75:
                        # Preserve punctuation around the word
                        prefix = ""
                        suffix = ""
                        for char in word:
                            if char.isalpha():
                                break
                            prefix += char
                        for char in reversed(word):
                            if char.isalpha():
                                break
                            suffix = char + suffix
                        
                        # Replace with correct name, preserving punctuation
                        corrected_words.append(prefix + correct_name + suffix)
                        continue
            
            # No replacement - keep original word
            corrected_words.append(word)
        
        return " ".join(corrected_words)
    
    def generate_task_title(self, sentence: str) -> str:
        """Generate a clean, short task title from the sentence."""
        title = sentence
        
        # Remove leading filler phrases
        remove_patterns = [
            r'^(?:hi|hello|hey)\s+\w+,?\s*',
            r'^(?:also|and|oh|one more thing|by the way),?\s*',
            r'^(?:we|you|someone|anybody)\s+(?:need|should|has|have)\s+(?:to\s+)?',
            r'^(?:i\s+think\s+)?(?:we|you)\s+(?:can|could|might)\s+',
            r'^\w+,\s*',  # Remove name at start: "Sakshi, we need..." -> "we need..."
        ]
        
        for pattern in remove_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Capitalize first letter
        title = title.strip()
        if title:
            title = title[0].upper() + title[1:]
        
        # Truncate if too long
        if len(title) > 60:
            title = title[:60].rsplit(' ', 1)[0] + '...'
        
        return title
    
    def uses_pronoun_reference(self, sentence: str) -> bool:
        """Check if sentence uses pronouns like 'this', 'that' to refer to previous context."""
        pronoun_patterns = [
            r'^we\s+should\s+tackle\s+this\b',
            r'\btackle\s+this\b',
            r'\bhandle\s+this\b',
            r'\bfix\s+this\b',
            r'\baddress\s+this\b',
        ]
        for pattern in pronoun_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False
    
    def extract_tasks(self, transcript: str, team_members: List[TeamMember]) -> List[Task]:
        """
        Main method: Extract all tasks from a meeting transcript.
        
        Pipeline:
        0. Normalize transcript (fix apostrophes, contractions, etc.)
        1. Split into sentences
        2. Identify task sentences (filter false positives)
        3. Carry context from filtered sentences (person mentions, topic keywords)
        4. Identify context sentences and merge with previous task
        5. Extract metadata (deadline, priority, person, dependencies)
        6. Create Task objects
        """
        print("\nExtracting tasks from transcript...")
        
        # Step 0: Normalize transcript (fix apostrophes, weird characters, etc.)
        transcript = normalize_transcript(transcript)
        print("Transcript normalized (fixed apostrophes, contractions, whitespace)")
        
        # Step 1: Split into sentences
        sentences = self.extract_sentences(transcript)
        print(f"Found {len(sentences)} sentences in transcript")
        
        tasks = []
        task_id = 1
        last_task = None  # Track the last task for context merging
        
        # Context from filtered sentences (for pronoun resolution)
        # When a sentence is filtered (e.g., question), save mentioned person and keywords
        pending_context: dict[str, Optional[str]] = {
            'person': None,
            'sentence': None,  # The filtered sentence text (for adding to description)
        }
        
        # Step 2: Process each sentence
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence) < 10:
                continue
            
            # Check if this is a context sentence (provides info for previous task)
            if self.is_context_sentence(sentence) and last_task is not None:
                # Merge context with the previous task
                print(f"  [Context] Merging with Task #{last_task.id}: {sentence[:40]}...")
                
                # Append to description
                last_task.description += " " + sentence
                
                # Extract additional metadata from context
                if not last_task.deadline:
                    last_task.deadline = self.extract_deadline(sentence)
                
                # Check for priority upgrades (pass deadline for urgency scoring)
                context_priority = self.extract_priority(sentence, last_task.deadline)
                # Priority hierarchy: Critical > High > Medium > Low
                # Upgrade if context suggests higher priority
                priority_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
                if priority_rank.get(context_priority, 0) > priority_rank.get(last_task.priority, 0):
                    last_task.priority = context_priority
                
                # Check for dependencies
                if self.has_dependency(sentence):
                    last_task._has_dependency = True
                
                continue
            
            # Check if this sentence describes a NEW task
            if not self.is_task_sentence(sentence):
                # Even though this isn't a task, save context (mentioned person, topic)
                # This helps when next sentence says "we should tackle this"
                mentioned = self.find_mentioned_person(sentence, team_members)
                if mentioned:
                    pending_context['person'] = mentioned
                    pending_context['sentence'] = sentence
                    print(f"  [Pending context] {mentioned} mentioned: {sentence[:40]}...")
                continue
            
            # Step 3: Extract task metadata
            deadline = self.extract_deadline(sentence)
            priority = self.extract_priority(sentence, deadline)  # Pass deadline for urgency scoring
            has_dep = self.has_dependency(sentence)
            mentioned_person = self.find_mentioned_person(sentence, team_members)
            
            # If no person mentioned but task uses pronoun AND we have pending context
            if not mentioned_person and pending_context['person']:
                if self.uses_pronoun_reference(sentence):
                    mentioned_person = pending_context['person']
                    print(f"  [Resolved] '{sentence[:30]}...' refers to {mentioned_person}")
            
            # Generate task title
            task_title = self.generate_task_title(sentence)
            
            # Build description - include pending context if it was used
            description = sentence
            if pending_context['sentence'] and mentioned_person == pending_context['person']:
                # Prepend the context sentence for clarity
                description = pending_context['sentence'] + " " + sentence
            
            # Clean up description - fix misspelled names
            if mentioned_person:
                description = self.correct_name_in_text(description, mentioned_person)
            
            # Step 4: Create Task object
            # Parse deadline to ISO format
            deadline_iso = self.parse_deadline_to_iso(deadline) if deadline else None
            
            task = Task(
                id=task_id,
                task=task_title,
                description=description,
                assigned_to=mentioned_person,
                deadline=deadline,
                deadline_iso=deadline_iso,
                priority=priority,
                dependencies=[],
                reason="Directly mentioned in meeting" if mentioned_person else None
            )
            
            task._has_dependency = has_dep
            
            tasks.append(task)
            last_task = task  # Track for context merging
            task_id += 1
            
            # Clear pending context after using it
            pending_context['person'] = None
            pending_context['sentence'] = None
            
            print(f"  Task {task.id}: {task_title[:50]}...")
        
        # Post-processing: Resolve dependencies and apply priority boosts
        print("\nPost-processing tasks...")
        self._resolve_dependencies_and_boost_priority(tasks)
        
        print(f"\nExtracted {len(tasks)} tasks from meeting")
        
        return tasks
    
    def _resolve_dependencies_and_boost_priority(self, tasks: List[Task]) -> None:
        """
        Post-processing step to:
        1. Resolve task dependencies (match dependency mentions to actual tasks)
        2. Boost priority for tasks that depend on Critical/High tasks
        
        This runs AFTER all tasks are extracted so we can see the full picture.
        """
        # Build a map of task IDs to priorities
        task_priorities = {task.id: task.priority for task in tasks}
        
        for task in tasks:
            # Check if this task has a dependency flag
            if not getattr(task, '_has_dependency', False):
                continue
            
            description_lower = task.description.lower()
            
            # Try to find which task it depends on
            for other_task in tasks:
                if other_task.id == task.id:
                    continue
                
                # Look for keywords from the other task in this task's description
                other_title_words = other_task.task.lower().split()
                
                # Count significant word matches
                matches = 0
                for word in other_title_words:
                    # Skip short/common words
                    if len(word) >= 4 and word in description_lower:
                        matches += 1
                
                # If enough words match, consider it a dependency
                if matches >= 2:
                    if other_task.id not in task.dependencies:
                        task.dependencies.append(other_task.id)
                        print(f"  Task #{task.id} depends on Task #{other_task.id}")
                    
                    # Apply priority boost if depending on Critical/High task
                    dep_priority = task_priorities.get(other_task.id, "Medium")
                    if dep_priority in ["Critical", "High"]:
                        # Re-calculate priority with dependency boost
                        new_priority = self.extract_priority(
                            task.description, 
                            task.deadline, 
                            has_critical_dependency=True
                        )
                        
                        # Only upgrade, never downgrade
                        priority_rank = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
                        if priority_rank.get(new_priority, 0) > priority_rank.get(task.priority, 0):
                            print(f"  Task #{task.id} priority boosted: {task.priority} → {new_priority} (depends on {dep_priority} task)")
                            task.priority = new_priority
                    
                    break  # Usually only one dependency mentioned


# Testing
if __name__ == "__main__":
    test_transcript = """
    Hi everyone, let's discuss this week's priorities.
    
    Sakshi, we need someone to fix the critical login bug that users reported yesterday. 
    This needs to be done by tomorrow evening since it's blocking users.
    
    Also, the database performance is really slow, Mohit you're good with backend optimization right?
    We should tackle this by end of this week, it's affecting the user experience.
    
    And we need to update the API documentation before Friday's release - this is high priority.
    
    Oh, and someone should design the new onboarding screens for the next sprint. 
    Arjun, didn't you work on UI designs last month? This can wait until next Monday.
    
    One more thing - we need to write unit tests for the payment module. 
    This depends on the login bug fix being completed first, so let's plan this for Wednesday.
    """
    
    from models import TeamMember
    team = [
        TeamMember("Sakshi", "Frontend Developer", ["React", "JavaScript", "UI bugs"]),
        TeamMember("Mohit", "Backend Engineer", ["Database", "APIs", "Performance optimization"]),
        TeamMember("Arjun", "UI/UX Designer", ["Figma", "User flows", "Mobile design"]),
        TeamMember("Lata", "QA Engineer", ["Testing", "Automation", "Quality assurance"]),
    ]
    
    extractor = TaskExtractor()
    tasks = extractor.extract_tasks(test_transcript, team)
    
    print("\n" + "="*60)
    print("EXTRACTED TASKS:")
    print("="*60)
    for task in tasks:
        print(f"\n#{task.id}: {task.task}")
        print(f"   Description: {task.description[:80]}...")
        print(f"   Deadline: {task.deadline or 'Not specified'}")
        print(f"   Priority: {task.priority}")
        print(f"   Mentioned: {task.assigned_to or 'None'}")
