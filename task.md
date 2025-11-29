# Project: Meeting Task Assignment

## Overview

Build a system that processes audio recordings of meetings and automatically assigns tasks to team members based on the meeting content.

## Problem Statement

In team meetings, multiple tasks are discussed and need to be assigned to appropriate team members. Currently, this process is manual and time-consuming. The goal is to automate task identification, prioritization, and assignment from meeting discussions.

## Example Scenario

### Meeting Context

A product development team has a weekly standup meeting discussing their mobile app project.

### Input:

### Audio Recording Contains:

"Hi everyone, let's discuss this week's priorities.

Sakshi, we need someone to fix the critical login bug that users reported yesterday. This needs to be done by tomorrow evening since it's blocking users.

Also, the database performance is really slow, Mohit you're good with backend optimization right?

We should tackle this by end of this week, it's affecting the user experience.

And we need to update the API documentation before Friday's release - this is high priority.

Oh, and someone should design the new onboarding screens for the next sprint. Arjun, didn't you work on UI designs last month? This can wait until next Monday.

One more thing - we need to write unit tests for the payment module. This depends on the login bug fix being completed first, so let's plan this for Wednesday."

### Team Members

| Name | Role | Skills |
| --- | --- | --- |
| Sakshi | Frontend Developer | React, JavaScript, UI bugs |
| Mohit | Backend Engineer | Database, APIs, Performance optimization |
| Arjun | UI/UX Designer | Figma, User flows, Mobile design |
| Lata | QA Engineer | Testing, Automation, Quality assurance |

---

## Expected Output:

### Identified Tasks with Details:

| # | Task | Description | Assigned To | Deadline | Priority | Dependencies (Optional) | Reason (Optional) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Fix critical login bug | Sakshi | Tomorrow evening | Critical / High | — | Frontend task, blocking users |  |
| 2 | Optimize database performance | Mohit | End of this week | High | — | Backend expertise |  |
| 3 | Update API documentation | Mohit | Friday | High | — | Backend related, before release |  |
| 4 | Design new onboarding screens | Arjun | Next Monday | Medium | — | UI/UX task, relevant experience |  |
| 5 | Write unit tests for payment module | Lata | Wednesday | Medium | Depends on Task #1 | QA expertise, testing task |  |

---

## Input Requirements

Your system should accept:

### 1. Audio File

- Format: `.wav`, `.mp3`, or `.m4a`
- Contains a recorded team meeting with task discussions

### 2. Team Member Information

- List of available team members
- Role and skill details for assignment logic

---

## Output Requirements

Your system must produce:

### Minimum Required:

### 1. List of Identified Tasks

- Each task extracted from the meeting
- Clear description of what needs to be done

### 2. Task Assignments

- Which team member is assigned to which task
- (Optional) Include reasoning if mentioned or inferred

### Additional Extractable Information:

- Deadlines / Timeline (e.g., “tomorrow”, “next week”, “by Friday”)
- Priority Levels (Critical, High, Medium, Low)
- Dependencies (Optional) such as “depends on Task #1”
- Contextual Notes (Optional) such as blockers or special conditions

---

## Technical Constraints

### ALLOWED:

- Use any external API or service for Speech-to-Text conversion (e.g., Whisper, Google STT)
- Use any library for parsing, NLP, or logic building
- You can train your own model or your logic for task identification and assignment

### NOT ALLOWED:

- No external APIs or pre-trained models for task classification or assignment logic
- No external automation tools that perform the core decision-making

**Note:**

The Speech-to-Text step can use external services, but identifying and assigning tasks must be your own logic or model.

---

## Deliverables

### 1. Source Code

- Clearly organized file structure
- Include a `requirements.txt` file

### 2. Test Data (Optional)

- Sample audio file(s)
- Sample team member data
- Expected output JSON or table

### 3. Demo Video

- Short walkthrough explaining your logic and result

---

## Submission Guidelines

Submit via GitHub Repository Link

Include clear README and instructions to run the project

---

## Questions?

If anything is unclear or you need further clarification, feel free to ask before starting. We’re happy to guide you through the process.

**Contact:** +916352617754 (WhatsApp only)

Good luck! We’re excited to see your solution.