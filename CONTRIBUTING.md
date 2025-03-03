# Contributing to EV Opinion Search Engine

Thank you for considering contributing to this project. Below are guidelines we'd like you to follow.

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Use docstrings for all functions and classes
- Keep functions small and focused on a single task
- Use type hints where appropriate

## Git Workflow

1. Create a branch for your feature or bugfix
2. Make your changes
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Project Structure

- Keep the modular structure intact
- Add new data sources in the `crawler` module
- Add new classification methods in the `classification` module
- Add new search features in the `indexing` module
- Update the web UI in the `web` module

## Testing

- Write unit tests for all new functionality
- Ensure existing tests pass before submitting changes