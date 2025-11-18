# ADK Test Project

Google ADK (Agent Development Kit) test project for learning and experimentation with AI agents.

## 🎯 What's Inside

This repository contains examples of AI agents built with Google's Agent Development Kit:

1. **Simple Time Agent** (`my_agent/`) - Basic agent with a mock tool
2. **Tutorial Agent** (`day1_agent_tutorial.py`) - Automated demo with Google Search
3. **Interactive Agent** (`interactive_agent.py`) - Chat-style agent in terminal

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/wtrout187/adktest1.git
cd adktest1

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install google-adk
pip install google-adk
```

### 2. Configure API Key

Create a `.env` file in the `my_agent/` folder:

```bash
echo 'GOOGLE_API_KEY="YOUR_API_KEY_HERE"' > my_agent/.env
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## 💻 Usage Examples

### Option 1: Web UI (Visual Interface)

```bash
adk web
```

Then open: http://127.0.0.1:8000

### Option 2: Interactive Terminal Chat

```bash
python interactive_agent.py
```

Ask questions and get real-time answers powered by Google Search!

### Option 3: Automated Demo

```bash
python day1_agent_tutorial.py
```

Runs 3 pre-scripted questions to demonstrate agent capabilities.

## 🔧 What Can These Agents Do?

- 🔍 **Search the web** for current information
- 🤔 **Think** about what tools to use
- 🎯 **Take action** by calling functions
- 💬 **Respond** with accurate, up-to-date answers

## 📚 Learning Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Getting Started Guide](https://google.github.io/adk-docs/get-started/python/)
- [Kaggle 5 Days of AI Course](https://www.kaggle.com/learn/5-days-of-ai)

## 🔐 Security

- ✅ API keys are stored in `.env` files (ignored by git)
- ✅ `.gitignore` configured to prevent credential leaks
- ⚠️ Never commit your `.env` file or API keys!

## 🛠️ Technologies

- **Python 3.12+**
- **Google ADK** - Agent Development Kit
- **Gemini 2.5 Flash** - Google's LLM
- **Google Search Tool** - Real-time web search

## 📝 Project Structure

```
ADKTest/
├── my_agent/              # Simple time agent
│   ├── agent.py          # Agent definition
│   └── .env              # API key (DO NOT COMMIT)
├── day1_agent_tutorial.py # Automated demo
├── interactive_agent.py   # Interactive chat
├── .gitignore            # Security configuration
└── README.md             # This file
```

## 🤝 Contributing

This is a learning project. Feel free to fork and experiment!

## 📄 License

Open source for educational purposes.

---

**Built with ❤️ using Google ADK**
