# Contributing to PaSSER

First off, thank you for considering contributing to PaSSER! It's people like you that make PaSSER such a great tool for the AI research community.

## Code of Conduct

This project and everyone participating in it is governed by our commitment to fostering an open and welcoming environment. We expect all contributors to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem
* **Describe the exact steps to reproduce the problem** in as much detail as possible
* **Provide specific examples** to demonstrate the steps
* **Describe the behavior you observed** after following the steps
* **Explain which behavior you expected to see instead** and why
* **Include screenshots and animated GIFs** if possible
* **Include your environment details**: OS, Node.js version, Python version, browser

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior** and explain which behavior you expected to see instead
* **Explain why this enhancement would be useful** to most PaSSER users
* **List some other applications where this enhancement exists**, if applicable

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the JavaScript and Python style guides
* Include thoughtfully-worded, well-structured tests
* Document new code based on the Documentation Style Guide
* End all files with a newline

## Development Process

### Setting Up Your Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/passer.git
   cd passer
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/passer.git
   ```

4. Install dependencies:
   ```bash
   npm install
   cd scripts
   pip install -r requirements.txt
   ```

5. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Making Changes

1. Make your changes in your feature branch
2. Add or update tests as needed
3. Ensure all tests pass:
   ```bash
   npm test
   ```

4. Update documentation if you changed APIs or added features
5. Commit your changes using clear commit messages:
   ```bash
   git commit -m "feat: add new RAG metric calculation"
   ```

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

* `feat`: A new feature
* `fix`: A bug fix
* `docs`: Documentation only changes
* `style`: Changes that don't affect the meaning of the code
* `refactor`: A code change that neither fixes a bug nor adds a feature
* `perf`: A code change that improves performance
* `test`: Adding missing tests or correcting existing tests
* `chore`: Changes to the build process or auxiliary tools

Examples:
```
feat: add BLEU score calculation for multi-agent tests
fix: resolve ChromaDB connection timeout issue
docs: update installation instructions for Windows
```

### Submitting Your Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request from your fork to the main repository
3. Fill in the PR template completely
4. Link any related issues
5. Wait for review and address any feedback

## Style Guides

### JavaScript Style Guide

* Use ES6+ features
* Use 2 spaces for indentation
* Use semicolons
* Use meaningful variable names
* Comment complex logic
* Follow React best practices
* Use functional components with hooks

Example:
```javascript
// Good
const calculateMetrics = (prediction, reference) => {
  const bleuScore = calculateBLEU(prediction, reference);
  const rougeScore = calculateROUGE(prediction, reference);
  
  return {
    bleu: bleuScore,
    rouge: rougeScore
  };
};

// Bad
function calc(p,r){
var b=calcB(p,r)
var ro=calcR(p,r)
return {b:b,ro:ro}}
```

### Python Style Guide

* Follow PEP 8
* Use 4 spaces for indentation
* Use type hints where appropriate
* Write docstrings for functions and classes
* Use meaningful variable names
* Keep functions focused and small

Example:
```python
# Good
def calculate_bleu_score(prediction: str, reference: str) -> float:
    """
    Calculate BLEU score between prediction and reference text.
    
    Args:
        prediction: The predicted text from the model
        reference: The ground truth reference text
        
    Returns:
        BLEU score as a float between 0.0 and 1.0
    """
    # Implementation here
    pass

# Bad
def calc(p,r):
    # calc bleu
    pass
```

### Documentation Style Guide

* Use Markdown for documentation
* Keep line length under 100 characters for readability
* Use code blocks with language specification
* Include examples where appropriate
* Link to related documentation
* Update table of contents when adding new sections

## Testing

### Frontend Tests

```bash
npm test
```

### Backend Tests

```bash
cd scripts
python -m pytest
```

### Integration Tests

Ensure your changes work with:
* Multiple LLM backends (Ollama, OpenAI)
* Different database configurations
* Various browser environments

## Project Structure

Understanding the project structure will help you navigate and contribute:

```
passer/
├── public/              # Static assets
├── src/
│   ├── component/       # React components
│   │   ├── backEnd.py   # Main RAG backend
│   │   ├── maBackEnd.py # Multi-agent backend
│   │   └── *.js         # React component files
│   ├── App.js           # Main application component
│   └── index.js         # Application entry point
├── scripts/             # Utility scripts
│   ├── requirements.txt # Python dependencies
│   └── *.py            # Backend scripts
├── maTests/             # Test results directory
└── package.json         # Node.js dependencies
```

## Adding New Features

### Adding a New Metric

1. Implement the metric calculation in `scripts/backEnd.py`
2. Add the metric to the frontend display in relevant components
3. Update documentation in README.md
4. Add tests for the new metric
5. Update the metrics table in documentation

### Adding a New LLM Integration

1. Add endpoint configuration in `configuration.json`
2. Implement the integration in backend services
3. Update the model selection UI
4. Test with various prompts and contexts
5. Document the integration process

### Adding a New Test Type

1. Create the test interface component
2. Implement backend endpoints for test execution
3. Add result storage and retrieval
4. Create visualization for results
5. Update documentation

## Questions?

Feel free to:
* Open an issue with the question label
* Contact the maintainers
* Check existing documentation and issues

## Recognition

Contributors will be:
* Listed in the CONTRIBUTORS.md file
* Mentioned in release notes
* Credited in academic publications when appropriate

Thank you for contributing to PaSSER! 🎉
