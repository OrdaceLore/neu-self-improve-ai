"""
WebShop Environment - Improved Simulation
Based on: "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agent"

This is a more realistic simulation of the WebShop environment that captures
the key challenges of the task. In production, this would connect to the
actual WebShop server.
"""

import torch
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class WebShopAction(Enum):
    """Actions available in WebShop"""
    SEARCH = 0      # Search for products
    CLICK = 1       # Click on an element
    SCROLL = 2      # Scroll page
    SELECT = 3      # Select option (size, color)
    BUY = 4         # Purchase item


@dataclass
class Product:
    """Simulated product in the catalog"""
    name: str
    price: float
    rating: float
    attributes: Dict[str, str]  # color, size, etc.
    asin: str  # Product ID
    
    def matches_query(self, query: str) -> float:
        """Check how well product matches search query (0-1)"""
        query_lower = query.lower()
        name_lower = self.name.lower()
        
        # Simple keyword matching
        keywords = query_lower.split()
        matches = sum(1 for kw in keywords if kw in name_lower)
        return matches / max(len(keywords), 1)


@dataclass
class WebShopTask:
    """A shopping task to complete"""
    instruction: str
    target_price: Optional[float] = None  # Max price constraint
    target_rating: Optional[float] = None  # Min rating constraint
    required_attributes: Optional[Dict[str, str]] = None  # Must have these attrs
    
    def evaluate_purchase(self, product: Product) -> float:
        """Evaluate how well a purchase satisfies the task (0-1)"""
        score = 0.5  # Base score for completing purchase
        
        # Price constraint
        if self.target_price:
            if product.price <= self.target_price:
                score += 0.2
            else:
                score -= 0.3  # Penalty for over-budget
        
        # Rating constraint
        if self.target_rating:
            if product.rating >= self.target_rating:
                score += 0.2
            else:
                score -= 0.1
        
        # Attribute matching
        if self.required_attributes:
            attr_matches = sum(
                1 for k, v in self.required_attributes.items()
                if product.attributes.get(k, "").lower() == v.lower()
            )
            score += 0.1 * attr_matches / len(self.required_attributes)
        
        return max(0, min(1, score))


class WebShopEnvironment:
    """
    More realistic WebShop environment simulation.
    
    Key features:
    - Product catalog with attributes
    - Search functionality
    - Multi-step navigation
    - Task-based evaluation
    """
    
    def __init__(self, catalog_size: int = 100):
        self.catalog = self._generate_catalog(catalog_size)
        self.tasks = self._generate_tasks()
        self.current_task: Optional[WebShopTask] = None
        self.current_page: str = "home"  # home, search_results, product_detail
        self.search_results: List[Product] = []
        self.selected_product: Optional[Product] = None
        self.selected_options: Dict[str, str] = {}
        self.steps_taken: int = 0
        self.max_steps: int = 20
        
    def _generate_catalog(self, size: int) -> List[Product]:
        """Generate simulated product catalog"""
        categories = ["shirt", "pants", "jacket", "shoes", "hat", "bag", "watch"]
        colors = ["black", "white", "blue", "red", "green", "gray"]
        sizes = ["S", "M", "L", "XL"]
        
        products = []
        for i in range(size):
            category = random.choice(categories)
            color = random.choice(colors)
            size = random.choice(sizes)
            
            product = Product(
                name=f"{color.capitalize()} {category} - Style {i}",
                price=round(random.uniform(10, 200), 2),
                rating=round(random.uniform(2.5, 5.0), 1),
                attributes={"color": color, "size": size, "category": category},
                asin=f"B{i:08d}"
            )
            products.append(product)
        
        return products
    
    def _generate_tasks(self) -> List[WebShopTask]:
        """Generate shopping tasks"""
        return [
            WebShopTask(
                instruction="Find a blue shirt under $50 with good rating",
                target_price=50.0,
                target_rating=4.0,
                required_attributes={"color": "blue", "category": "shirt"}
            ),
            WebShopTask(
                instruction="Buy any black jacket",
                required_attributes={"color": "black", "category": "jacket"}
            ),
            WebShopTask(
                instruction="Find the cheapest shoes available",
                required_attributes={"category": "shoes"}
            ),
            WebShopTask(
                instruction="Get a highly rated watch (4.5+ stars)",
                target_rating=4.5,
                required_attributes={"category": "watch"}
            ),
            WebShopTask(
                instruction="Buy a red bag under $30",
                target_price=30.0,
                required_attributes={"color": "red", "category": "bag"}
            ),
        ]
    
    def reset(self) -> torch.Tensor:
        """Reset environment with new task"""
        self.current_task = random.choice(self.tasks)
        self.current_page = "home"
        self.search_results = []
        self.selected_product = None
        self.selected_options = {}
        self.steps_taken = 0
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Get current state representation.
        
        State encodes:
        - Current page (one-hot: 3 values)
        - Number of search results (normalized)
        - Selected product features (if any)
        - Task completion progress
        - Steps remaining (normalized)
        """
        # Page encoding (one-hot)
        page_enc = [0.0, 0.0, 0.0]
        if self.current_page == "home":
            page_enc[0] = 1.0
        elif self.current_page == "search_results":
            page_enc[1] = 1.0
        else:  # product_detail
            page_enc[2] = 1.0
        
        # Search results count (normalized)
        results_count = len(self.search_results) / 20.0
        
        # Selected product features
        if self.selected_product:
            price_norm = self.selected_product.price / 200.0
            rating_norm = self.selected_product.rating / 5.0
            product_features = [price_norm, rating_norm]
        else:
            product_features = [0.0, 0.0]
        
        # Steps remaining
        steps_remaining = (self.max_steps - self.steps_taken) / self.max_steps
        
        # Has selected product (binary)
        has_product = 1.0 if self.selected_product else 0.0
        
        state = page_enc + [results_count] + product_features + [steps_remaining, has_product]
        return torch.tensor(state, dtype=torch.float32)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take action in environment.
        
        Actions:
        0: Search (uses task keywords)
        1: Click (select top result or detail)
        2: Scroll (get more results)
        3: Select option
        4: Buy
        """
        self.steps_taken += 1
        reward = -0.05  # Small step penalty
        done = False
        info = {}
        
        if action == WebShopAction.SEARCH.value:
            # Perform search
            if self.current_task:
                # Extract keywords from instruction
                keywords = self.current_task.instruction.lower()
                self.search_results = [
                    p for p in self.catalog
                    if p.matches_query(keywords) > 0.2
                ][:10]
                self.current_page = "search_results"
                reward = 0.1 if self.search_results else 0.0
        
        elif action == WebShopAction.CLICK.value:
            if self.current_page == "search_results" and self.search_results:
                # Select best matching product
                self.selected_product = self.search_results[0]
                self.current_page = "product_detail"
                reward = 0.1
            elif self.current_page == "home":
                reward = -0.1  # Invalid action
        
        elif action == WebShopAction.SCROLL.value:
            if self.current_page == "search_results":
                # Could add more results, small reward
                reward = 0.01
        
        elif action == WebShopAction.SELECT.value:
            if self.current_page == "product_detail" and self.selected_product:
                # Auto-select matching options
                reward = 0.05
        
        elif action == WebShopAction.BUY.value:
            if self.current_page == "product_detail" and self.selected_product:
                # Complete purchase - evaluate success
                task_score = self.current_task.evaluate_purchase(self.selected_product)
                reward = task_score
                done = True
                info["success"] = task_score > 0.5
                info["task_score"] = task_score
            else:
                reward = -0.2  # Tried to buy without selecting product
        
        # Check step limit
        if self.steps_taken >= self.max_steps:
            done = True
            info["timeout"] = True
        
        return self._get_state(), reward, done, info
    
    @property
    def state_dim(self) -> int:
        return 8
    
    @property
    def action_dim(self) -> int:
        return 5


# Alias for backward compatibility
MockWebShop = WebShopEnvironment

