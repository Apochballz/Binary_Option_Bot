// Configuration
const API_BASE_URL = window.location.origin + '/api';
let selectedPair = 'ALL';
let autoRefreshInterval;
let countdownInterval;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Binary Options Dashboard Loaded');
    loadPairs();
    loadSignals();
    updateTimes();
    checkMarketStatus();
    startAutoRefresh();
    
    // Update clocks every second
    setInterval(updateTimes, 1000);
    
    // Check market status every minute
    setInterval(checkMarketStatus, 60000);
});

// Update time displays
function updateTimes() {
    const now = new Date();
    
    // Local time
    const localTime = now.toLocaleTimeString();
    document.getElementById('localTime').textContent = localTime;
    
    // UTC time
    const utcHours = String(now.getUTCHours()).padStart(2, '0');
    const utcMinutes = String(now.getUTCMinutes()).padStart(2, '0');
    const utcSeconds = String(now.getUTCSeconds()).padStart(2, '0');
    document.getElementById('utcTime').textContent = `${utcHours}:${utcMinutes}:${utcSeconds}`;
    
    // Trading session
    const utcHour = now.getUTCHours();
    let session = '';
    
    if (utcHour >= 12 && utcHour < 15) {
        session = '🔥 OVERLAP';
    } else if (utcHour >= 7 && utcHour < 12) {
        session = '🇬🇧 LONDON';
    } else if (utcHour >= 13 && utcHour < 17) {
        session = '🇺🇸 NY';
    } else {
        session = '😴 OFF-HOURS';
    }
    
    document.getElementById('tradingSession').textContent = session;
}

// Load currency pairs
function loadPairs() {
    const pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'EURGBP', 'GBPJPY'];
    const pairsSelector = document.getElementById('pairsSelector');
    
    let html = '<button class="pair-btn active" onclick="selectPair(\'ALL\')">ALL PAIRS</button>';
    
    pairs.forEach(pair => {
        html += `<button class="pair-btn" onclick="selectPair('${pair}')">${pair}</button>`;
    });
    
    pairsSelector.innerHTML = html;
}

// Select currency pair
function selectPair(pair) {
    selectedPair = pair;
    
    // Update active button
    document.querySelectorAll('.pair-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Reload signals
    loadSignals();
}

// Load signals from API
async function loadSignals() {
    const container = document.getElementById('signalsContainer');
    container.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading signals...</p></div>';
    
    try {
        const url = selectedPair === 'ALL' 
            ? `${API_BASE_URL}/signals/active`
            : `${API_BASE_URL}/signals/active?pair=${selectedPair}`;
        
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.success && data.signals && data.signals.length > 0) {
            displaySignals(data.signals);
            
            // Update win rate if available
            if (data.win_rate) {
                document.getElementById('winRate').textContent = `${data.win_rate.toFixed(1)}%`;
            }
        } else {
            container.innerHTML = `
                <div class="no-signals">
                    <h2>🔍 No active signals</h2>
                    <p>Waiting for high-confidence setups (75%+)</p>
                    <p>Market conditions are being analyzed...</p>
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading signals:', error);
        container.innerHTML = `
            <div class="no-signals">
                <h2>❌ Connection Error</h2>
                <p>Unable to fetch signals from backend.</p>
                <p>Please check if the server is running.</p>
                <button onclick="loadSignals()" style="margin-top: 20px; padding: 10px 20px; background: white; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem;">
                    🔄 Retry
                </button>
            </div>
        `;
    }
}

// Display signals
function displaySignals(signals) {
    const container = document.getElementById('signalsContainer');
    
    const html = signals.map(signal => {
        const directionClass = signal.direction === 'CALL' ? 'call' : 'put';
        const directionEmoji = signal.direction === 'CALL' ? '🟢' : '🔴';
        const directionText = signal.direction === 'CALL' ? 'BUY CALL' : 'SELL PUT';
        
        // Build metrics HTML
        let metricsHtml = `
            <div class="metric-box">
                <strong>Entry Price</strong>
                <span>${signal.entry_price.toFixed(5)}</span>
            </div>
        `;
        
        if (signal.indicators) {
            metricsHtml += `
                <div class="metric-box">
                    <strong>RSI</strong>
                    <span>${signal.indicators.rsi || '--'}</span>
                </div>
                <div class="metric-box">
                    <strong>ADX</strong>
                    <span>${signal.indicators.adx || '--'}</span>
                </div>
                <div class="metric-box">
                    <strong>ATR %</strong>
                    <span>${signal.indicators.atr_percent || '--'}%</span>
                </div>
            `;
        }
        
        if (signal.risk_level) {
            metricsHtml += `
                <div class="metric-box">
                    <strong>Risk Level</strong>
                    <span>${signal.risk_level}</span>
                </div>
            `;
        }
        
        if (signal.regime && signal.regime.regime) {
            metricsHtml += `
                <div class="metric-box">
                    <strong>Market Regime</strong>
                    <span>${signal.regime.regime}</span>
                </div>
            `;
        }
        
        if (signal.confluence && signal.confluence.score) {
            metricsHtml += `
                <div class="metric-box">
                    <strong>Confluence</strong>
                    <span>${signal.confluence.score.toFixed(1)}%</span>
                </div>
            `;
        }
        
        if (signal.position_sizing && signal.position_sizing.position_size) {
            metricsHtml += `
                <div class="metric-box">
                    <strong>Suggested Stake</strong>
                    <span>$${signal.position_sizing.position_size.toFixed(0)}</span>
                </div>
            `;
        }
        
        // AI Insight
        let aiInsightHtml = '';
        if (signal.ai_insight) {
            aiInsightHtml = `
                <div class="ai-insight-box">
                    <strong>🤖 AI Insight:</strong>
                    <span>${signal.ai_insight}</span>
                </div>
            `;
        }
        
        return `
            <div class="signal-card">
                <div class="signal-header">
                    <div class="signal-pair">${signal.pair_name || signal.pair}</div>
                    <div class="signal-confidence">${signal.confidence}%</div>
                </div>
                
                <div class="signal-direction ${directionClass}">
                    <span>${directionEmoji}</span>
                    <span>${directionText}</span>
                </div>
                
                <div class="metrics-grid">
                    ${metricsHtml}
                </div>
                
                <div class="timing-info">
                    <div>
                        <strong>Entry Time (Local)</strong>
                        <span>${formatLocalTime(signal.entry_time)}</span>
                    </div>
                    <div>
                        <strong>Expiry Time</strong>
                        <span>${formatLocalTime(signal.expiry_time)}</span>
                    </div>
                    <div>
                        <strong>Time Remaining</strong>
                        <span class="countdown" data-expiry="${signal.expiry_timestamp}">--:--</span>
                    </div>
                </div>
                
                ${aiInsightHtml}
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
    
    // Start countdown timers
    startCountdowns();
}

// Format time to local timezone
function formatLocalTime(utcTimeStr) {
    if (!utcTimeStr) return '--:--';
    
    try {
        // Parse UTC time (format: HH:MM UTC)
        const timeMatch = utcTimeStr.match(/(\d{2}):(\d{2})/);
        if (!timeMatch) return utcTimeStr;
        
        const [, hours, minutes] = timeMatch;
        
        // Create date object with today's date and the time
        const utcDate = new Date();
        utcDate.setUTCHours(parseInt(hours), parseInt(minutes), 0, 0);
        
        // Convert to local time
        return utcDate.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit'
        });
    } catch (e) {
        console.error('Error formatting time:', e);
        return utcTimeStr;
    }
}

// Start countdown timers
function startCountdowns() {
    // Clear existing interval
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }
    
    countdownInterval = setInterval(() => {
        const countdownElements = document.querySelectorAll('.countdown');
        
        countdownElements.forEach(el => {
            const expiryISO = el.dataset.expiry;
            if (!expiryISO) return;
            
            try {
                const expiry = new Date(expiryISO);
                const now = new Date();
                const diff = expiry - now;
                
                if (diff <= 0) {
                    el.textContent = 'EXPIRED';
                    el.style.color = '#ef4444';
                } else {
                    const minutes = Math.floor(diff / 60000);
                    const seconds = Math.floor((diff % 60000) / 1000);
                    el.textContent = `${minutes}:${String(seconds).padStart(2, '0')}`;
                    
                    // Change color based on time remaining
                    if (minutes === 0 && seconds <= 30) {
                        el.style.color = '#ef4444'; // Red
                    } else if (minutes <= 1) {
                        el.style.color = '#f59e0b'; // Orange
                    } else {
                        el.style.color = '#78350f'; // Default
                    }
                }
            } catch (e) {
                console.error('Error parsing expiry time:', e);
                el.textContent = 'ERROR';
            }
        });
    }, 1000);
}

// Check market status
async function checkMarketStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/market/status`);
        const data = await response.json();
        
        const statusEl = document.getElementById('marketStatus');
        
        if (data.is_open) {
            statusEl.textContent = '● LIVE';
            statusEl.style.color = '#4ade80';
            statusEl.classList.add('status-live');
        } else {
            statusEl.textContent = '● CLOSED';
            statusEl.style.color = '#ef4444';
            statusEl.classList.remove('status-live');
        }
    } catch (error) {
        console.error('Market status check failed:', error);
        const statusEl = document.getElementById('marketStatus');
        statusEl.textContent = '● UNKNOWN';
        statusEl.style.color = '#fbbf24';
    }
}

// Auto-refresh signals every 30 seconds
function startAutoRefresh() {
    autoRefreshInterval = setInterval(() => {
        console.log('🔄 Auto-refreshing signals...');
        loadSignals();
    }, 30000); // 30 seconds
}

// Manual refresh
function refreshSignals() {
    console.log('🔄 Manual refresh triggered');
    loadSignals();
}

// Stop auto-refresh (if needed)
function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
    }
    if (countdownInterval) {
        clearInterval(countdownInterval);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopAutoRefresh();
});