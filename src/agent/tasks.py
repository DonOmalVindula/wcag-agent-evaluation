"""
Predefined web tasks for evaluation.

Each task defines a URL, description, and success criteria.
Tasks are grouped by sector and difficulty.
"""

from .task_runner import WebTask

# Navigation tasks — test basic page traversal and link following
NAVIGATION_TASKS = [
    WebTask(
        url="https://www.bbc.com",
        description="Navigate to the Sport section from the homepage",
        success_criteria="The page URL contains '/sport' or the page heading mentions Sport",
        max_steps=5,
    ),
    WebTask(
        url="https://www.gov.uk",
        description="Find the page about renewing a passport",
        success_criteria="The page contains information about passport renewal",
        max_steps=8,
    ),
    WebTask(
        url="https://www.who.int",
        description="Navigate to the page about COVID-19",
        success_criteria="The page contains information about COVID-19 or coronavirus",
        max_steps=8,
    ),
]

# Search tasks — test search functionality interaction
SEARCH_TASKS = [
    WebTask(
        url="https://www.bbc.com",
        description="Use the search function to search for 'climate change'",
        success_criteria="Search results for 'climate change' are displayed",
        max_steps=8,
    ),
    WebTask(
        url="https://www.coursera.org",
        description="Search for 'machine learning' courses",
        success_criteria="Search results showing machine learning courses are displayed",
        max_steps=8,
    ),
]

# Information extraction tasks — test reading and understanding page content
INFO_TASKS = [
    WebTask(
        url="https://www.nhs.uk",
        description="Find information about symptoms of flu",
        success_criteria="The page displays flu symptoms information",
        max_steps=10,
    ),
]

# All tasks combined
ALL_TASKS = NAVIGATION_TASKS + SEARCH_TASKS + INFO_TASKS
