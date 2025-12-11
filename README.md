# Binary_Option_Bot
Option Trading Bot Signal 
# 🎯 Binary Options AI Signal Generator

Professional binary options trading signal system with 80% target win rate using AI + Technical Analysis + Order Flow.

## 🚀 Features

- ✅ **Market Regime Detection** - Trending/Ranging/Volatile classification
- ✅ **Confluence Scoring** - Multi-indicator alignment (70%+ required)
- ✅ **Session-Based Trading** - Only trades during peak liquidity (London/NY)
- ✅ **Adaptive Risk Management** - ATR-based dynamic stop-loss
- ✅ **Kelly Criterion Position Sizing** - Optimal stake calculation
- ✅ **Order Flow Integration** - Bid/Ask imbalance + CVD tracking
- ✅ **Real-time Dashboard** - User location-aware timing
- ✅ **VectorBT Backtesting** - Professional performance analysis

## 📊 Performance Targets

| Phase | Timeline | Win Rate | Key Features |
|-------|----------|----------|--------------|
| Phase 1 | 1-3 months | 68-72% | Regime + Confluence + Session |
| Phase 2 | 3-6 months | 72-76% | + Adaptive SL + Position Sizing |
| Phase 3 | 6-12 months | 78-82% | + Real Order Flow (OANDA) |

## 🛠️ Tech Stack

- **Backend**: Flask (Python 3.11)
- **Analysis**: TA-Lib, VectorBT, SquareQuant
- **AI**: DeepSeek API
- **Data**: Twelve Data / OANDA
- **Deployment**: Railway
- **Frontend**: Vanilla JS (Single Page)

## 📦 Local Installation

### Prerequisites
- Python 3.11+
- pip
- TA-Lib (system library)

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/binary-options-bot.git
cd binary-options-bot
```

### Step 2: Install TA-Lib (System Dependency)

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
```

**Windows:**
Download pre-built: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### Step 3: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys
nano .env
```

### Step 5: Run Application
```bash
python app.py
```

Visit: http://localhost:5000

## 🚂 Railway Deployment

### Option A: Deploy via CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# Set environment variables
railway variables set TWELVE_DATA_KEY=xxx
railway variables set DEEPSEEK_API_KEY=xxx
railway variables set MINIMUM_CONFIDENCE=75

# Open deployed app
railway open
```

### Option B: Deploy via Dashboard
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose this repository
5. Add environment variables in dashboard
6. Railway auto-deploys on every push

## 📡 API Endpoints

### Generate Signal
```http
POST /api/signals/generate
Content-Type: application/json

{
  "symbol": "EURUSD"
}

Response:
{
  "success": true,
  "signal": {
    "direction": "CALL",
    "confidence": 82.5,
    "entry_price": 1.10245,
    "regime": "TRENDING",
    "confluence_score": 78.5
  }
}
```

### Get Active Signals
```http
GET /api/signals/active?pair=EURUSD

Response:
{
  "success": true,
  "signals": [...],
  "win_rate": 78.5
}
```

### Market Status
```http
GET /api/market/status

Response:
{
  "is_open": true,
  "session": "OVERLAP"
}
```

### Run Backtest
```http
POST /api/vectorbt/backtest
Content-Type: application/json

{
  "pair": "EURUSD",
  "days": 30
}

Response:
{
  "success": true,
  "results": {
    "total_return": 45.2,
    "sharpe_ratio": 2.1,
    "win_rate": 78.5
  }
}
```

## 📊 Signal Filtering Logic
```
┌─────────────────────────────────────────┐
│ FILTER 1: Session Check                │
│ ✅ London/NY overlap (12:00-15:00 UTC) │
│ ❌ Off-hours, weekends, Friday close   │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│ FILTER 2: Regime Detection             │
│ ✅ Trending (ADX > 25)                 │
│ ❌ Ranging (ADX < 20) - SKIP           │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│ FILTER 3: Confluence Scoring           │
│ ✅ Score > 70%                         │
│ ❌ Score < 70% - SKIP                  │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│ FILTER 4: Order Flow Confirmation      │
│ ✅ Imbalance > 1.3 or < 0.77           │
│ ⚠️  Neutral (0.8-1.2) - Reduce conf   │
└────────────┬────────────────────────────┘
             │
┌────────────▼────────────────────────────┐
│ FINAL CHECK: Confidence > 75%          │
│ ✅ Generate signal                     │
│ ❌ Reject and wait                     │
└─────────────────────────────────────────┘
```

## 🎯 Trading Rules

### ✅ GREEN LIGHTS (High Probability)
- Confluence > 80%
- London/NY overlap (12:00-15:00 UTC)
- Trending market (ADX > 25)
- Order flow aligned (strong imbalance)
- All indicators agree

### ❌ RED FLAGS (Always Skip)
- ADX < 20 (ranging market)
- Confluence < 70%
- Off-hours trading
- Friday after 15:00 UTC
- Weekends
- Extreme volatility (ATR% > 1.5%)

## 📈 Performance Metrics

Expected after 6 months optimization:
- **Win Rate**: 78-82%
- **Profit Factor**: 2.5+
- **Sharpe Ratio**: 2.0+
- **Max Drawdown**: <10%
- **Signals/Day**: 5-8
- **Avg Confidence**: 81%

## 🔐 Security

- All API keys in environment variables
- CORS properly configured
- Rate limiting on endpoints
- Input validation
- HTTPS enforced in production
- No sensitive data in logs

## 📝 License

MIT License - See LICENSE file for details

## ⚠️ Disclaimer

**Risk Warning**: Binary options trading involves significant risk of loss. This system is for educational purposes only. Past performance does not guarantee future results. Never risk more than you can afford to lose.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📞 Support

- **Issues**: https://github.com/YOUR_USERNAME/binary-options-bot/issues
- **Discussions**: https://github.com/YOUR_USERNAME/binary-options-bot/discussions
- **Email**: support@yourdomain.com

## 🙏 Acknowledgments

- TA-Lib for technical indicators
- VectorBT for backtesting framework
- DeepSeek for AI analysis
- Railway for hosting platform

---

**Built with ❤️ for traders who take signals seriously**
