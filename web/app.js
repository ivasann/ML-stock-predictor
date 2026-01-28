/**
 * ML Analytics Suite - Clean Fintech Dashboard
 * Interactive JavaScript for Stock, Sales, and Churn Analysis
 */

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initEventListeners();
    initAnimations();
    initStockCards();
    initLiveFeed();
});

// ========== CHARTS ==========
let mainChart, salesChart;

function initCharts() {
    initMainChart();
    initSalesChart();
}

function initMainChart() {
    const ctx = document.getElementById('mainChart');
    if (!ctx) return;

    const labels = generateDateLabels(60);
    const historical = generateStockData(45, 75, 92);
    const predicted = generateStockData(15, 88, 98);

    mainChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical',
                    data: [...historical, ...Array(15).fill(null)],
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                },
                {
                    label: 'Predicted',
                    data: [...Array(44).fill(null), historical[44], ...predicted],
                    borderColor: '#22C55E',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#000',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => ctx.raw ? `₹${ctx.raw.toFixed(2)}` : null
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#737373', font: { family: 'Space Grotesk', size: 11 }, maxTicksLimit: 8 }
                },
                y: {
                    grid: { color: '#EBEBEB' },
                    ticks: { color: '#737373', font: { family: 'Space Grotesk', size: 11 }, callback: v => `₹${v}` }
                }
            }
        }
    });
}

function initSalesChart() {
    const ctx = document.getElementById('salesChart');
    if (!ctx) return;

    salesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Food', 'Drinks', 'Non-Consumable'],
            datasets: [
                {
                    label: 'Actual',
                    data: [4520000, 1280000, 2340000],
                    backgroundColor: '#BFFF00',
                    borderRadius: 6,
                },
                {
                    label: 'Predicted',
                    data: [4680000, 1350000, 2420000],
                    backgroundColor: '#22C55E',
                    borderRadius: 6,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { color: '#737373', font: { family: 'Space Grotesk', size: 12 }, usePointStyle: true }
                },
                tooltip: {
                    backgroundColor: '#000',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ₹${(ctx.raw / 100000).toFixed(1)}L`
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#737373', font: { family: 'Space Grotesk' } } },
                y: {
                    grid: { color: '#EBEBEB' },
                    ticks: { color: '#737373', font: { family: 'Space Grotesk' }, callback: v => `₹${(v / 100000).toFixed(0)}L` }
                }
            }
        }
    });
}

// ========== DATA GENERATORS ==========
function generateDateLabels(days) {
    const labels = [];
    const today = new Date();
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i + 15);
        labels.push(date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' }));
    }
    return labels;
}

function generateStockData(count, min, max) {
    const data = [];
    let value = min + (max - min) / 2;
    for (let i = 0; i < count; i++) {
        const change = (Math.random() - 0.45) * 2.5;
        value = Math.max(min, Math.min(max, value + change));
        data.push(parseFloat(value.toFixed(2)));
    }
    return data;
}

// ========== EVENT LISTENERS ==========
function initEventListeners() {
    // Fetch button
    document.getElementById('fetch-btn')?.addEventListener('click', () => {
        showLoading('Fetching stock data...');
        setTimeout(() => {
            hideLoading();
            showNotification('Stock data loaded successfully!');
        }, 1500);
    });

    // Predict button
    document.getElementById('predict-btn')?.addEventListener('click', () => {
        showLoading('Training LSTM model...');
        setTimeout(() => {
            updatePrediction();
            hideLoading();
            showNotification('Prediction complete! +7.72% expected');
        }, 2500);
    });

    // Chart tabs
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
        });
    });

    // Nav links smooth scroll
    document.querySelectorAll('.nav-link[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

// ========== STOCK CARDS ==========
function initStockCards() {
    document.querySelectorAll('.stock-card').forEach(card => {
        card.addEventListener('click', () => {
            const stock = card.dataset.stock;
            document.getElementById('stock-select').value = stock;
            showNotification(`Selected ${stock}`);
        });
    });
}

// ========== PREDICTIONS ==========
function updatePrediction() {
    const current = 87.45;
    const predicted = current * (1 + (Math.random() * 0.1 + 0.02));
    const change = ((predicted / current) - 1) * 100;

    // Update forecast display
    const forecastValue = document.querySelector('.forecast-value.highlight');
    if (forecastValue) {
        forecastValue.textContent = `₹${predicted.toFixed(2)}`;
    }

    const badge = document.querySelector('.forecast-badge');
    if (badge) {
        badge.textContent = `↗ +${change.toFixed(2)}%`;
        badge.className = 'forecast-badge positive';
    }

    // Update chart
    if (mainChart) {
        const historical = generateStockData(45, 75, 92);
        const predicted = generateStockData(15, 88, 102);
        mainChart.data.datasets[0].data = [...historical, ...Array(15).fill(null)];
        mainChart.data.datasets[1].data = [...Array(44).fill(null), historical[44], ...predicted];
        mainChart.update();
    }
}

// ========== ANIMATIONS ==========
function initAnimations() {
    // Scroll Reveal implementation
    const observerOptions = {
        threshold: 0.15,
        rootMargin: '0px 0px -50px 0px'
    };

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                revealObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Apply reveal to sections and cards
    document.querySelectorAll('section, .stock-card, .btn-cta, .hero-title-group, .live-feed-container').forEach(el => {
        el.classList.add('reveal');
        revealObserver.observe(el);
    });

    // Animate mini chart bars
    document.querySelectorAll('.mini-chart .bar').forEach((bar, i) => {
        bar.style.animationDelay = `${i * 50}ms`;
    });

    // Start advanced features
    initParallax();
    startVolatility();
}

// Mouse Parallax Effect for Hero Cards
function initParallax() {
    const hero = document.querySelector('.hero');
    const cards = document.querySelectorAll('.stat-card, .portfolio-card');

    if (!hero || cards.length === 0) return;

    hero.addEventListener('mousemove', (e) => {
        const x = (e.clientX / window.innerWidth - 0.5) * 40;
        const y = (e.clientY / window.innerHeight - 0.5) * 40;

        cards.forEach((card, index) => {
            const factor = (index + 1) * 0.5;
            card.style.transform = `translate(${x * factor}px, ${y * factor}px)`;
        });
    });

    hero.addEventListener('mouseleave', () => {
        cards.forEach(card => {
            card.style.transform = 'translate(0, 0)';
        });
    });
}

// Mimic real-time volatility with small chart "twitches"
function startVolatility() {
    setInterval(() => {
        const bars = document.querySelectorAll('.mini-chart .bar');
        if (bars.length === 0) return;

        // Randomly pick 2-3 bars to twitch
        for (let i = 0; i < 3; i++) {
            const randomBar = bars[Math.floor(Math.random() * bars.length)];
            const currentHeight = parseFloat(randomBar.style.height || '50%');
            const twitch = (Math.random() - 0.5) * 10;
            const newHeight = Math.max(20, Math.min(80, currentHeight + twitch));
            randomBar.style.height = `${newHeight}%`;
        }
    }, 800);
}

// ========== UI HELPERS ==========
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading');
    if (overlay) {
        overlay.querySelector('.loader-text').textContent = message;
        overlay.classList.add('active');
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading');
    if (overlay) overlay.classList.remove('active');
}

function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = `<span class="notif-icon">✓</span> ${message}`;

    Object.assign(notification.style, {
        position: 'fixed',
        bottom: '24px',
        right: '24px',
        padding: '16px 24px',
        background: '#000',
        color: '#fff',
        borderRadius: '8px',
        fontWeight: '600',
        fontSize: '14px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
        zIndex: '3000',
        animation: 'slideUp 0.3s ease',
        fontFamily: 'Space Grotesk, sans-serif'
    });

    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        .notif-icon {
            width: 20px;
            height: 20px;
            background: #BFFF00;
            color: #000;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }
    `;
    document.head.appendChild(style);
    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(20px)';
        notification.style.transition = 'all 0.3s';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Initialize counters
document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count);
    let current = 0;
    const increment = target / 50;
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            el.textContent = target.toLocaleString();
            clearInterval(timer);
        } else {
            el.textContent = Math.floor(current).toLocaleString();
        }
    }, 30);
});

// ========== LIVE FEED & CHAT ==========
let newsStats = { bullish: 0, bearish: 0, neutral: 0, total: 0 };
let currentSentiment = 65;
let isPaused = false;

const newsEvents = [
    { source: 'reuters', text: 'Global demand for AI chips surges 25% in Q1; tech sector leads rally.', sentiment: 'bullish', stocks: ['TCS', 'INFY', 'WIPRO'], impact: 1.5 },
    { source: 'et', text: 'RBI signals potential rate cut if inflation stays below 4%; markets jump.', sentiment: 'bullish', stocks: ['RELIANCE', 'SUBEX'], impact: 2.1 },
    { source: 'cnbc', text: 'Crude oil prices spike 4% amid Middle East tensions; pressure on OMCs.', sentiment: 'bearish', stocks: ['RELIANCE'], impact: -1.2 },
    { source: 'bloomberg', text: 'US Federal Reserve holds rates steady; signals "wait and watch" approach.', sentiment: 'neutral', stocks: ['TCS', 'INFY'], impact: 0.2 },
    { source: 'moneycontrol', text: 'Subex wins major contract for 5G fraud analytics in Southeast Asia.', sentiment: 'bullish', stocks: ['SUBEX'], impact: 4.8 },
    { source: 'et', text: 'Corporate tax cuts expected in upcoming budget; industry sentiment high.', sentiment: 'bullish', stocks: ['RELIANCE', 'TCS'], impact: 1.8 },
    { source: 'reuters', text: 'Major security flaw found in popular cloud platform; IT stocks slip.', sentiment: 'bearish', stocks: ['INFY', 'WIPRO', 'TCS'], impact: -2.4 },
    { source: 'cnbc', text: 'Global logistics slowed by port strikes; supply chain concerns rise.', sentiment: 'bearish', stocks: ['RELIANCE'], impact: -0.8 },
    { source: 'moneycontrol', text: 'Wipro revenue guidance for Q3 beats analyst expectations.', sentiment: 'bullish', stocks: ['WIPRO'], impact: 2.5 },
    { source: 'bloomberg', text: 'European markets open flat ahead of ECB meeting minutes.', sentiment: 'neutral', stocks: [], impact: 0 },
    { source: 'reuters', text: 'Japan Yen hits record low; yen-denominated exports surge.', sentiment: 'neutral', stocks: ['TCS'], impact: 0.3 },
    { source: 'et', text: 'India manufacturing PMI at 3-year high as factory orders pour in.', sentiment: 'bullish', stocks: ['RELIANCE', 'SUBEX'], impact: 1.9 }
];

function initLiveFeed() {
    // Add event listeners for chat controls
    document.getElementById('pause-feed')?.addEventListener('click', () => {
        isPaused = !isPaused;
        document.getElementById('pause-feed').textContent = isPaused ? '▶️' : '⏸️';
        showNotification(isPaused ? 'Live feed paused' : 'Live feed resumed');
    });

    document.getElementById('clear-feed')?.addEventListener('click', () => {
        const chat = document.getElementById('news-chat');
        if (chat) chat.innerHTML = '';
        showNotification('Chat feed cleared');
    });

    // Start news stream
    setInterval(() => {
        if (!isPaused) postRandomNews();
    }, 4000 + Math.random() * 4000);

    // Periodic price fluctuations
    setInterval(() => {
        if (!isPaused) fluctuatePrices();
    }, 3000);

    // Initial news
    postRandomNews();
    setTimeout(postRandomNews, 1500);
}

function postRandomNews() {
    const event = newsEvents[Math.floor(Math.random() * newsEvents.length)];
    addChatMessage(event);
    updateSentiment(event.sentiment);

    // Apply news impact to stocks
    event.stocks.forEach(stock => {
        const impact = event.impact;
        updateStockPrice(stock, impact);
    });
}

function addChatMessage(event) {
    const chat = document.getElementById('news-chat');
    if (!chat) return;

    const msg = document.createElement('div');
    msg.className = 'chat-message';

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const sentimentEmoji = event.sentiment === 'bullish' ? '📈' : (event.sentiment === 'bearish' ? '📉' : '➡️');

    const stocksHtml = event.stocks.map(s => {
        const isUp = event.sentiment === 'bullish' || (event.sentiment === 'neutral' && Math.random() > 0.5);
        return `<span class="msg-stock-tag ${isUp ? 'up' : 'down'}">${s} ${isUp ? '+' : ''}${event.impact}%</span>`;
    }).join('');

    msg.innerHTML = `
        <div class="msg-header">
            <span class="msg-source ${event.source}">${event.source}</span>
            <span class="msg-time">${time}</span>
            <span class="msg-sentiment">${sentimentEmoji}</span>
        </div>
        <p class="msg-text">${event.text}</p>
        <div class="msg-stocks">${stocksHtml}</div>
    `;

    chat.appendChild(msg);
    chat.scrollTop = chat.scrollHeight;

    // Remove old messages if too many (keep last 20)
    if (chat.childNodes.length > 20) {
        chat.removeChild(chat.firstChild);
    }

    // Update stats
    newsStats[event.sentiment]++;
    newsStats.total++;
    updateStatsUI();
}

function updateStockPrice(stockId, changePercent) {
    const priceEl = document.getElementById(`price-${stockId}`);
    const changeEl = document.getElementById(`change-${stockId}`);
    const flashEl = document.getElementById(`flash-${stockId}`);

    if (!priceEl || !changeEl) return;

    // Current price extract (handle ₹ and commas)
    let currentPrice = parseFloat(priceEl.textContent.replace(/[₹,]/g, ''));

    // Calculate new price
    const change = (currentPrice * (changePercent / 100)) + (Math.random() - 0.5) * 2;
    currentPrice += change;

    // Update UI
    priceEl.textContent = `₹${currentPrice.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const actualChangePercent = (change / (currentPrice - change)) * 100;
    const isPositive = actualChangePercent >= 0;

    changeEl.textContent = `${isPositive ? '+' : ''}${actualChangePercent.toFixed(2)}%`;
    changeEl.className = `stock-change ${isPositive ? 'positive' : 'negative'}`;

    // Flash animation
    if (flashEl) {
        flashEl.className = `stock-flash ${isPositive ? 'flash-green' : 'flash-red'}`;
        setTimeout(() => {
            flashEl.className = 'stock-flash';
        }, 500);
    }
}

function fluctuatePrices() {
    const stocks = ['SUBEX', 'TCS', 'INFY', 'RELIANCE', 'WIPRO'];
    stocks.forEach(stock => {
        // Small random walk +/- 0.1%
        const walk = (Math.random() - 0.5) * 0.2;
        updateStockPrice(stock, walk);
    });
}

function updateSentiment(newSentiment) {
    if (newSentiment === 'bullish') currentSentiment += 2;
    if (newSentiment === 'bearish') currentSentiment -= 2;

    currentSentiment = Math.max(15, Math.min(95, currentSentiment));

    const valueEl = document.getElementById('sentiment-value');
    const ringEl = document.getElementById('sentiment-ring');
    const fillEl = document.getElementById('mood-fill');

    if (valueEl) valueEl.textContent = Math.round(currentSentiment);

    // Update SVG stroke-dashoffset (251.2 is full circle)
    if (ringEl) {
        const offset = 251.2 - (251.2 * (currentSentiment / 100));
        ringEl.style.strokeDashoffset = offset;
        ringEl.style.stroke = currentSentiment > 50 ? '#BFFF00' : '#EF4444';
    }

    // Update Mood Bar
    if (fillEl) {
        fillEl.style.left = `${currentSentiment}%`;
    }

    const moodText = document.getElementById('mood-text');
    if (moodText) {
        if (currentSentiment > 60) moodText.textContent = 'Bullish 📈';
        else if (currentSentiment < 40) moodText.textContent = 'Bearish 📉';
        else moodText.textContent = 'Neutral ➡️';
    }
}

function updateStatsUI() {
    const bCount = document.getElementById('bullish-count');
    const beCount = document.getElementById('bearish-count');
    const nCount = document.getElementById('neutral-count');
    const tCount = document.getElementById('total-news');

    if (bCount) bCount.textContent = newsStats.bullish;
    if (beCount) beCount.textContent = newsStats.bearish;
    if (nCount) nCount.textContent = newsStats.neutral;
    if (tCount) tCount.textContent = `${newsStats.total} news today`;

    // Update breakdown bars
    const total = newsStats.total;
    if (total > 0) {
        const pPct = (newsStats.bullish / total) * 100;
        const nPct = (newsStats.bearish / total) * 100;
        const neuPct = (newsStats.neutral / total) * 100;

        const pBar = document.getElementById('positive-bar');
        const nBar = document.getElementById('negative-bar');
        const neuBar = document.getElementById('neutral-bar');

        if (pBar) pBar.style.width = `${pPct}%`;
        if (nBar) nBar.style.width = `${nPct}%`;
        if (neuBar) neuBar.style.width = `${neuPct}%`;

        document.getElementById('positive-pct').textContent = `${Math.round(pPct)}%`;
        document.getElementById('negative-pct').textContent = `${Math.round(nPct)}%`;
        document.getElementById('neutral-pct').textContent = `${Math.round(neuPct)}%`;
    }
}
