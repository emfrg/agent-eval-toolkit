# Contributing to AI Agent Conversation Simulator

Thank you for your interest in contributing! This project is designed to be extensible and we welcome contributions of all kinds.

## How to Contribute

### 1. Report Issues

Found a bug or have a feature request? Please open an issue on GitHub with:
- Clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (OS, Python version, DeepEval version)

### 2. Submit Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test your changes**
   ```bash
   llm-evals-starter quickstart
   ```
5. **Commit with clear messages**
   ```bash
   git commit -m "Add: Description of your changes"
   ```
6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### 3. Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings for all public functions/classes
- Keep functions focused and modular

**Format code:**
```bash
pip install black ruff
black .
ruff check .
```

## What to Contribute

### Easy Wins (Great for First-Time Contributors)

- **Add agent adapters** - Examples for Anthropic, Cohere, Hugging Face, etc.
- **Add custom metrics** - Create domain-specific evaluation metrics:
  - Code-based metrics (conversation length, response time, etc.)
  - LLM-judge metrics with custom rubrics (tone, empathy, domain accuracy)
  - See [src/metrics/custom_examples.py](src/metrics/custom_examples.py) for templates
- **Improve documentation** - Fix typos, add examples, clarify instructions
- **Add tests** - Unit tests for core functionality
- **Fix bugs** - Check the issues page

### Medium Difficulty

- **Enhance existing metrics** - Improve ConversationLengthMetric, OnboardingClarityMetric
- **Add advanced metrics** - Multi-turn consistency, topic tracking, sentiment analysis
- **Improve CLI** - Add new commands or options
- **Enhance reporting** - Better visualizations, export formats
- **Add examples** - Real-world use cases and tutorials

### Advanced

- **Web UI** - Dashboard for configuration and visualization
- **Database integration** - Store logs and results in PostgreSQL/MongoDB
- **Real-time monitoring** - Stream simulation progress
- **Multi-agent evaluation** - Compare multiple agent versions

## Project Structure

```
src/
├── core/           # Core types and utilities
├── simulation/     # Conversation simulation
├── evaluation/     # LLM-as-a-judge and reporting
├── metrics/        # Custom evaluation metrics
│   ├── base.py            # Helper utilities
│   └── custom_examples.py # Example metrics
└── cli.py          # Command-line interface

examples/           # Agent adapter and usage examples
config/             # Configuration files
```

### Adding a Custom Metric

1. **Create your metric** in `src/metrics/custom_examples.py` or a new file:

```python
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import ConversationalTestCase

class MyMetric(BaseMetric):
    def __init__(self, threshold=0.5, name="My Metric"):
        self.threshold = threshold
        self.name = name

    def measure(self, test_case, _show_indicator=True):
        # Your evaluation logic
        self.score = 0.0  # Calculate score
        self.reason = "Explanation"
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return self.name
```

2. **Export it** in `src/metrics/__init__.py`
3. **Add example usage** in `examples/custom_metrics_example.py`
4. **Update README** with brief description
5. **Submit PR** with clear description of what the metric evaluates

## Development Setup

```bash
# Clone your fork
git clone https://github.com/emfrg/llm-evals-starter.git
cd llm-evals-starter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .
pip install -e ".[dev]"

# Run tests
python -m pytest

# Run quickstart to verify
llm-evals-starter quickstart
```

## Testing

Before submitting a PR, ensure:

1. **Code runs without errors**
   ```bash
   llm-evals-starter validate-config config/personas.yaml
   llm-evals-starter quickstart
   ```

2. **No breaking changes** (unless discussed in issue)

3. **Documentation updated** (if adding features)

## Questions?

- Open a GitHub issue
- Tag maintainers in discussions
- Check existing issues/PRs for similar topics

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the problem, not the person
- Help others learn and grow

---

**Thank you for contributing!** 🎉
