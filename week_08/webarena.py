"""
WebArena Environment - Improved Simulation
Based on: "WebArena: A Realistic Web Environment for Building Autonomous Agents"

This provides a more realistic simulation of WebArena tasks, capturing the
key challenges of multi-step web navigation.
"""

import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class WebArenaAction(Enum):
    """Actions available in WebArena"""
    CLICK = 0       # Click on element
    TYPE = 1        # Type text into field
    SCROLL = 2      # Scroll page
    SUBMIT = 3      # Submit form
    NAVIGATE = 4    # Navigate to URL
    SELECT = 5      # Select dropdown option


@dataclass
class WebElement:
    """Represents an HTML element"""
    tag: str
    id: Optional[str]
    text: str
    attributes: Dict[str, str]
    interactable: bool
    
    def __str__(self):
        return f"<{self.tag} id='{self.id}'>{self.text[:20]}...</{self.tag}>"


@dataclass
class WebPage:
    """Represents a web page state"""
    url: str
    title: str
    elements: List[WebElement]
    form_data: Dict[str, str]
    
    def get_interactable_elements(self) -> List[WebElement]:
        return [e for e in self.elements if e.interactable]


@dataclass
class WebArenaTask:
    """A web navigation task"""
    instruction: str
    website: str  # e.g., "shopping", "gitlab", "reddit"
    required_actions: List[str]  # Sequence of required action types
    success_criteria: str  # What determines success
    
    def check_completion(self, action_history: List[str], final_page: WebPage) -> float:
        """Check task completion (0-1)"""
        # Check if required action sequence was followed
        action_match = 0
        for required in self.required_actions:
            if required in action_history:
                action_match += 1
        
        action_score = action_match / max(len(self.required_actions), 1)
        
        # Check success criteria (simplified)
        success_score = 0.0
        if "login" in self.success_criteria and "logged_in" in str(final_page.url):
            success_score = 1.0
        elif "search" in self.success_criteria and "results" in str(final_page.title):
            success_score = 1.0
        elif "form" in self.success_criteria and final_page.form_data:
            success_score = 0.8
        
        return 0.4 * action_score + 0.6 * success_score


class WebArenaEnvironment:
    """
    More realistic WebArena environment simulation.
    
    Features:
    - Multiple website types (shopping, gitlab, reddit-like)
    - Multi-step task sequences
    - Form interactions
    - Navigation state tracking
    """
    
    def __init__(self):
        self.tasks = self._generate_tasks()
        self.current_task: Optional[WebArenaTask] = None
        self.current_page: Optional[WebPage] = None
        self.action_history: List[str] = []
        self.steps_taken: int = 0
        self.max_steps: int = 30
        
        # Website templates
        self.websites = self._generate_websites()
    
    def _generate_tasks(self) -> List[WebArenaTask]:
        """Generate realistic web tasks"""
        return [
            WebArenaTask(
                instruction="Log into the shopping website and search for laptops",
                website="shopping",
                required_actions=["navigate", "type", "click", "type", "submit"],
                success_criteria="search_results"
            ),
            WebArenaTask(
                instruction="Create a new issue on GitLab titled 'Bug report'",
                website="gitlab",
                required_actions=["navigate", "click", "type", "submit"],
                success_criteria="form_submitted"
            ),
            WebArenaTask(
                instruction="Post a comment on the first post in the forum",
                website="reddit",
                required_actions=["navigate", "click", "type", "submit"],
                success_criteria="comment_posted"
            ),
            WebArenaTask(
                instruction="Change the profile name to 'Test User'",
                website="shopping",
                required_actions=["navigate", "click", "type", "submit"],
                success_criteria="profile_updated"
            ),
            WebArenaTask(
                instruction="Add an item to cart and proceed to checkout",
                website="shopping",
                required_actions=["navigate", "click", "click", "navigate"],
                success_criteria="checkout_page"
            ),
        ]
    
    def _generate_websites(self) -> Dict[str, List[WebPage]]:
        """Generate website page templates"""
        return {
            "shopping": [
                WebPage(
                    url="https://shop.example.com/",
                    title="Shop - Home",
                    elements=[
                        WebElement("input", "search", "", {"type": "text", "placeholder": "Search..."}, True),
                        WebElement("button", "search-btn", "Search", {}, True),
                        WebElement("a", "login", "Login", {"href": "/login"}, True),
                        WebElement("a", "cart", "Cart (0)", {"href": "/cart"}, True),
                    ],
                    form_data={}
                ),
                WebPage(
                    url="https://shop.example.com/login",
                    title="Shop - Login",
                    elements=[
                        WebElement("input", "username", "", {"type": "text"}, True),
                        WebElement("input", "password", "", {"type": "password"}, True),
                        WebElement("button", "login-submit", "Login", {}, True),
                    ],
                    form_data={}
                ),
                WebPage(
                    url="https://shop.example.com/results",
                    title="Shop - Search Results",
                    elements=[
                        WebElement("div", "product-1", "Laptop - $999", {}, True),
                        WebElement("div", "product-2", "Phone - $599", {}, True),
                        WebElement("button", "add-cart", "Add to Cart", {}, True),
                    ],
                    form_data={}
                ),
            ],
            "gitlab": [
                WebPage(
                    url="https://gitlab.example.com/",
                    title="GitLab - Home",
                    elements=[
                        WebElement("a", "new-issue", "New Issue", {"href": "/issues/new"}, True),
                        WebElement("a", "projects", "Projects", {"href": "/projects"}, True),
                    ],
                    form_data={}
                ),
                WebPage(
                    url="https://gitlab.example.com/issues/new",
                    title="GitLab - New Issue",
                    elements=[
                        WebElement("input", "title", "", {"placeholder": "Issue title"}, True),
                        WebElement("textarea", "description", "", {}, True),
                        WebElement("button", "submit", "Create Issue", {}, True),
                    ],
                    form_data={}
                ),
            ],
            "reddit": [
                WebPage(
                    url="https://reddit.example.com/",
                    title="Reddit - Home",
                    elements=[
                        WebElement("div", "post-1", "First Post Title", {}, True),
                        WebElement("textarea", "comment", "", {"placeholder": "Write a comment..."}, True),
                        WebElement("button", "submit-comment", "Comment", {}, True),
                    ],
                    form_data={}
                ),
            ],
        }
    
    def reset(self) -> torch.Tensor:
        """Reset environment with new task"""
        self.current_task = random.choice(self.tasks)
        website = self.current_task.website
        self.current_page = self.websites[website][0]  # Start at home page
        self.action_history = []
        self.steps_taken = 0
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Get current state representation.
        
        Encodes:
        - Current page type
        - Number of interactable elements
        - Task progress (actions completed)
        - Steps remaining
        """
        if not self.current_page:
            return torch.zeros(10)
        
        # Page type encoding (3 types)
        page_type = [0.0, 0.0, 0.0]
        if "home" in self.current_page.title.lower():
            page_type[0] = 1.0
        elif "login" in self.current_page.title.lower() or "new" in self.current_page.title.lower():
            page_type[1] = 1.0
        else:
            page_type[2] = 1.0
        
        # Number of interactable elements (normalized)
        n_elements = len(self.current_page.get_interactable_elements()) / 10.0
        
        # Task progress
        if self.current_task:
            progress = len(self.action_history) / max(len(self.current_task.required_actions), 1)
        else:
            progress = 0.0
        
        # Steps remaining
        steps_remaining = (self.max_steps - self.steps_taken) / self.max_steps
        
        # Form data status
        has_form_data = 1.0 if self.current_page.form_data else 0.0
        
        # Website type (3 types)
        website_enc = [0.0, 0.0, 0.0]
        if self.current_task:
            if self.current_task.website == "shopping":
                website_enc[0] = 1.0
            elif self.current_task.website == "gitlab":
                website_enc[1] = 1.0
            else:
                website_enc[2] = 1.0
        
        state = page_type + [n_elements, progress, steps_remaining, has_form_data] + website_enc
        return torch.tensor(state, dtype=torch.float32)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take action in environment"""
        self.steps_taken += 1
        reward = -0.02  # Small step penalty
        done = False
        info = {}
        
        action_name = WebArenaAction(action).name.lower()
        self.action_history.append(action_name)
        
        if action == WebArenaAction.CLICK.value:
            # Click action - might navigate or interact
            elements = self.current_page.get_interactable_elements()
            if elements:
                element = random.choice(elements)
                if element.tag == "a":
                    # Navigation link clicked
                    self._navigate_to_next_page()
                    reward = 0.1
                elif element.tag == "button":
                    reward = 0.15
        
        elif action == WebArenaAction.TYPE.value:
            # Type action - fill form field
            elements = [e for e in self.current_page.elements if e.tag in ["input", "textarea"]]
            if elements:
                element = elements[0]
                self.current_page.form_data[element.id] = "test_input"
                reward = 0.1
        
        elif action == WebArenaAction.SUBMIT.value:
            # Submit form
            if self.current_page.form_data:
                # Form submitted - check task completion
                task_score = self.current_task.check_completion(
                    self.action_history, 
                    self.current_page
                )
                reward = task_score
                if task_score > 0.5:
                    done = True
                    info["success"] = True
        
        elif action == WebArenaAction.NAVIGATE.value:
            self._navigate_to_next_page()
            reward = 0.05
        
        elif action == WebArenaAction.SCROLL.value:
            reward = 0.01
        
        elif action == WebArenaAction.SELECT.value:
            reward = 0.05
        
        # Check step limit
        if self.steps_taken >= self.max_steps:
            done = True
            info["timeout"] = True
            # Partial credit for progress
            if self.current_task:
                final_score = self.current_task.check_completion(
                    self.action_history,
                    self.current_page
                )
                reward += final_score * 0.3
        
        return self._get_state(), reward, done, info
    
    def _navigate_to_next_page(self):
        """Navigate to next page in website"""
        if not self.current_task:
            return
        
        pages = self.websites.get(self.current_task.website, [])
        if len(pages) > 1:
            # Find next page in sequence
            current_idx = 0
            for i, page in enumerate(pages):
                if page.url == self.current_page.url:
                    current_idx = i
                    break
            
            next_idx = min(current_idx + 1, len(pages) - 1)
            self.current_page = pages[next_idx]
    
    @property
    def state_dim(self) -> int:
        return 10
    
    @property
    def action_dim(self) -> int:
        return 6


# Alias for backward compatibility
MockWebArena = WebArenaEnvironment

