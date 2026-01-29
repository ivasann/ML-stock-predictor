import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.stock_predictor import StockPredictor
from models.sales_forecaster import SalesForecaster
from models.churn_analyzer import ChurnAnalyzer
from config.settings import STOCK_SYMBOLS

app = FastAPI(title="ML Analytics Suite API")

class StockRequest(BaseModel):
    symbol: str
    period: str = "2y"
    predict_days: int = 30

@app.get("/")
async def read_index():
    return FileResponse("web/index.html")

@app.get("/api/stocks")
async def get_stocks():
    return STOCK_SYMBOLS

@app.post("/api/predict/stock")
async def predict_stock(req: StockRequest):
    try:
        predictor = StockPredictor(req.symbol)
        data = predictor.fetch_data(req.period)
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Prepare and train (simplified for web demo if no existing model)
        X_train, X_test, y_train, y_test = predictor.prepare_data()
        predictor.build_model()
        predictor.train(X_train, y_train, X_test, y_test, epochs=5) # Fewer epochs for speed
        
        # Predict
        future_dates, predictions = predictor.predict_future(req.predict_days)
        
        # Format results for Chart.js
        historical = data['Close'].tail(100).tolist()
        historical_dates = [d.strftime('%Y-%m-%d') for d in data.index[-100:]]
        
        return {
            "symbol": req.symbol,
            "historical": historical,
            "historical_dates": historical_dates,
            "predicted": predictions.tolist(),
            "predicted_dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "metrics": predictor.evaluate(X_test, y_test)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/sales")
async def predict_sales():
    try:
        forecaster = SalesForecaster()
        forecaster.generate_sample_data(2000)
        metrics = forecaster.train()
        forecast = forecaster.forecast_by_category()
        importance = forecaster.get_feature_importance().head(5).to_dict(orient='records')
        
        return {
            "metrics": metrics,
            "forecast": forecast.to_dict(orient='records'),
            "importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/churn")
async def predict_churn():
    try:
        analyzer = ChurnAnalyzer()
        analyzer.generate_sample_data(1000)
        metrics = analyzer.train()
        summary = analyzer.get_churn_summary()
        importance = analyzer.get_feature_importance().head(5).to_dict(orient='records')
        
        return {
            "metrics": metrics,
            "summary": summary,
            "importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news/social")
async def get_social_news():
    social_sources = [
        {"platform": "twitter", "user": "MarketWatch", "handle": "@MarketWatch", "verified": True},
        {"platform": "twitter", "user": "Elon Musk", "handle": "@elonmusk", "verified": True},
        {"platform": "instagram", "user": "FinShot", "handle": "finshots.in", "verified": True},
        {"platform": "stocktwits", "user": "ChartMill", "handle": "chartmill", "verified": False},
        {"platform": "twitter", "user": "NSE India", "handle": "@NSEIndia", "verified": True}
    ]
    
    news_templates = [
        {"text": "Breaking: {stock} showing strong reversal on daily charts. RSI oversold.", "sentiment": "bullish", "impact": 2.5},
        {"text": "Is {stock} the next big play for 2026? Q3 results expected to be stellar.", "sentiment": "bullish", "impact": 1.8},
        {"text": "Warning: Supply chain disruptions might hit {stock} margins in coming quarter.", "sentiment": "bearish", "impact": -1.2},
        {"text": "Institutional buying spotted in {stock} in last 3 trading sessions. Accumulation phase?", "sentiment": "bullish", "impact": 3.1},
        {"text": "Technical breakdown in {stock}. Support at ₹{price} broken. Watch out.", "sentiment": "bearish", "impact": -2.4}
    ]
    
    # Generate 5 fresh items
    items = []
    stocks = list(STOCK_SYMBOLS.keys())
    
    for _ in range(3):
        source = random.choice(social_sources)
        template = random.choice(news_templates)
        stock = random.choice(stocks)
        
        # Simple price mock for template
        price = random.randint(100, 3000)
        
        items.append({
            "id": random.randint(1000, 9999),
            "platform": source["platform"],
            "user": source["user"],
            "handle": source["handle"],
            "verified": source["verified"],
            "text": template["text"].format(stock=stock, price=price),
            "sentiment": template["sentiment"],
            "impact": template["impact"],
            "stocks": [stock],
            "timestamp": datetime.now().isoformat(),
            "engagement": {
                "likes": random.randint(10, 500),
                "reposts": random.randint(5, 100)
            }
        })
    
    return items

# Mount static files at the end
app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
