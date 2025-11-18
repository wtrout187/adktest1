# ADK Test Project

Google ADK (Agent Development Kit) test project for learning and experimentation.

## Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   
   # Windows CMD
   venv\Scripts\activate.bat
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install google-adk:**
   ```bash
   pip install google-adk
   ```

## Usage

Follow the official Google ADK documentation: https://google.github.io/adk-docs/get-started/python/

### Create an agent project
```bash
adk create my_agent
```

### Run your agent
```bash
adk run
```

### Web interface
```bash
adk web
```

## Environment Variables

Create a `.env` file in your project root with your API keys:
```
GOOGLE_API_KEY="YOUR_API_KEY"
```

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Getting Started Guide](https://google.github.io/adk-docs/get-started/python/)
