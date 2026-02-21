from langchain.messages import AIMessage
from datetime import datetime
import time

prices = [
    {
        "model": "gpt-5.2",
        "input": 1.75,          # USD pro 1M Input-Tokens
        "output": 14.0          # USD pro 1M Output-Tokens
    },
    {
        "model": "gpt-5.2-pro",
        "input": 21.0,
        "output": 168.0
    },
    {
        "model": "gpt-5.1",
        "input": 1.25,
        "output": 10.0
    },
    {
        "model": "gpt-5",
        "input": 1.25,
        "output": 10.0
    },
    {
        "model": "gpt-5-mini",
        "input": 0.25,
        "output": 2.0
    },
    {
        "model": "gpt-5-nano",
        "input": 0.05,
        "output": 0.40
    },
    {
        "model": "gpt-5-pro",
        "input": 15.0,
        "output": 120.0
    },
    {
        "model": "gpt-4.1",
        "input": 2.0,
        "output": 8.0
    },
    {
        "model": "gpt-4.1-mini",
        "input": 0.40,
        "output": 1.60
    },
    {
        "model": "gpt-4.1-nano",
        "input": 0.10,
        "output": 0.40
    },
    {
        "model": "gpt-4o",
        "input": 2.50,
        "output": 10.0
    },
    {
        "model": "gpt-4o-mini",
        "input": 0.15,
        "output": 0.60
    },
    {
        "model": "gpt-realtime",
        "input": 4.0,
        "output": 16.0
    },
    {
        "model": "gpt-realtime-mini",
        "input": 0.60,
        "output": 2.40
    }
]


# Base class for tracking token usage
class TokenUsage:
    def __init__(self, resp, model=None):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        self.prompt_USD = 0
        self.completion_USD = 0
        self.total_USD = 0

        self.timestamp = time.time()
        self.model = model
        
        self._extract_usage(resp)
        self.calc_USD()
    
    def _extract_usage(self, resp):
        pass
    
    def print(self):
        readable_time = datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')

        print("=" * 50)
        print("TOKEN USAGE STATISTICS")
        print("=" * 50)
        print(f"Model: {self.model or 'N/A'}")
        print(f"Prompt Tokens: {self.prompt_tokens}")
        print(f"Prompt USD: {self.prompt_USD}")
        print(f"Completion Tokens: {self.completion_tokens}")
        print(f"Completion USD: {self.completion_USD}")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"Total USD: {self.total_USD}")
        print(f"Timestamp: {readable_time}")
        print("=" * 50)

    def __str__(self):
        return f"Used Tokens:  Prompt_Tokens={self.prompt_tokens}={self.prompt_USD}$ Completion_Tokens={self.completion_tokens}={self.completion_USD}$ Total_Tokens={self.total_tokens}={self.total_USD}$"
    
    def calc_USD(self):
        for price in prices:
            if price["model"] == self.model:
                self.prompt_USD = (self.prompt_tokens / 1_000_000) * price["input"]
                self.completion_USD = (self.completion_tokens / 1_000_000) * price["output"]
                self.total_USD = self.prompt_USD + self.completion_USD

 

# Subclass for tracking token usage from OpenAI
class TokenUsageOpenAi(TokenUsage):
    def _extract_usage(self, resp):
        messages = resp.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Model Name
                if hasattr(msg, 'response_metadata') and msg.response_metadata:
                    if self.model is None:
                        self.model = msg.response_metadata.get('model_name')
                
                # Tokens
                if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                    self.prompt_tokens += msg.usage_metadata.get('input_tokens', 0)
                    self.completion_tokens += msg.usage_metadata.get('output_tokens', 0)
                    self.total_tokens += msg.usage_metadata.get('total_tokens', 0)


# Class for tracking hole token usage
class CostsTracker:

    def __init__(self):
        self.costsList = []
        self.out_dir = None


    def __del__(self):
        if self.out_dir is not None:
            with open(self.out_dir / "costs_log.csv", "a") as f:
                total = self.sum()
                f.write(f"SUMMARIZED,-,-,{total['prompt_tokens']},{total['prompt_USD']},{total['completion_tokens']},{total['completion_USD']},{total['total_tokens']},{total['total_USD']}\n")

    def add(self, cost):
        self.costsList.append(cost)

        if self.out_dir is not None:
            with open(self.out_dir / "costs_log.csv", "a") as f:
                f.write(f"{len(self.costsList)},{datetime.fromtimestamp(cost.timestamp).strftime('%Y-%m-%d %H:%M:%S')},{cost.model},{cost.prompt_tokens},{cost.prompt_USD},{cost.completion_tokens},{cost.completion_USD},{cost.total_tokens},{cost.total_USD}\n")
        
    def print(self):
        total = self.sum()
        print("=" * 50)
        print("TOTAL TOKEN USAGE STATISTICS")
        print("=" * 50)
        print(f"Prompt Tokens: {total['prompt_tokens']}")
        print(f"Prompt USD: {total['prompt_USD']}")
        print(f"Completion Tokens: {total['completion_tokens']}")
        print(f"Completion USD: {total['completion_USD']}")
        print(f"Total Tokens: {total['total_tokens']}")
        print(f"Total USD: {total['total_USD']}")
        print("=" * 50)

    def __str__(self):
        total = self.sum()
        return f"Total Used Tokens:  Prompt_Tokens={total['prompt_tokens']}={total['prompt_USD']}$ Completion_Tokens={total['completion_tokens']}={total['completion_USD']}$ Total_Tokens={total['total_tokens']}={total['total_USD']}$"

    def sum(self):
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "prompt_USD": 0,
            "completion_USD": 0,
            "total_USD": 0
        }
        for cost in self.costsList:
            total["prompt_tokens"] += cost.prompt_tokens
            total["completion_tokens"] += cost.completion_tokens
            total["total_tokens"] += cost.total_tokens
            total["prompt_USD"] += cost.prompt_USD
            total["completion_USD"] += cost.completion_USD
            total["total_USD"] += cost.total_USD
        return total

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir
        with open(self.out_dir / "costs_log.csv", "w") as f:
            f.write("Position,Timestamp,Model,Prompt Tokens,Prompt USD,Completion Tokens,Completion USD,Total Tokens,Total USD\n")


_COST_TRACKER = CostsTracker()


def get_cost_tracker():
    return _COST_TRACKER
