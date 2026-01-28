"""
ML Analytics Suite - Unified Dashboard Application
A comprehensive GUI for Stock Prediction, Sales Forecasting, and Churn Analysis
"""
import customtkinter as ctk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import COLORS, WINDOW_SIZE, STOCK_SYMBOLS
from models.stock_predictor import StockPredictor
from models.sales_forecaster import SalesForecaster
from models.churn_analyzer import ChurnAnalyzer
from utils.visualization import (
    setup_dark_style, create_stock_chart, create_forecast_chart,
    create_churn_gauge, create_feature_importance_chart
)


class MLAnalyticsSuite(ctk.CTk):
    """Main application window for ML Analytics Suite."""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("ML Analytics Suite - Indian Markets")
        self.geometry(WINDOW_SIZE)
        
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize models
        self.stock_predictor = None
        self.sales_forecaster = None
        self.churn_analyzer = None
        
        # Track chart canvases for cleanup
        self.current_canvases = []
        
        # Build UI
        self._create_header()
        self._create_tabview()
        self._create_status_bar()
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
    def _create_header(self):
        """Create application header."""
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS['card'])
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Logo/Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="📊 ML Analytics Suite",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=COLORS['primary']
        )
        title_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Stock Prediction • Sales Forecasting • Churn Analysis",
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        )
        subtitle_label.grid(row=0, column=1, padx=20, pady=15, sticky="w")
        
        # Timestamp
        self.time_label = ctk.CTkLabel(
            header_frame,
            text=datetime.now().strftime("%d %b %Y, %I:%M %p"),
            font=ctk.CTkFont(size=11),
            text_color="#666666"
        )
        self.time_label.grid(row=0, column=2, padx=20, pady=15, sticky="e")
        
    def _create_tabview(self):
        """Create main tabbed interface."""
        self.tabview = ctk.CTkTabview(
            self,
            fg_color=COLORS['background'],
            segmented_button_fg_color=COLORS['card'],
            segmented_button_selected_color=COLORS['primary'],
            segmented_button_unselected_color=COLORS['card']
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add tabs
        self.tab_stock = self.tabview.add("📈 Stock Predictor")
        self.tab_sales = self.tabview.add("🛒 Sales Forecast")
        self.tab_churn = self.tabview.add("👥 Churn Analyzer")
        
        # Configure tabs
        for tab in [self.tab_stock, self.tab_sales, self.tab_churn]:
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(1, weight=1)
        
        # Build tab contents
        self._build_stock_tab()
        self._build_sales_tab()
        self._build_churn_tab()
        
    def _build_stock_tab(self):
        """Build Stock Predictor tab content."""
        # Controls Frame
        controls_frame = ctk.CTkFrame(self.tab_stock, fg_color=COLORS['card'])
        controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        controls_frame.grid_columnconfigure(3, weight=1)
        
        # Stock selector
        ctk.CTkLabel(controls_frame, text="Stock:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=(15, 5), pady=15
        )
        
        self.stock_var = ctk.StringVar(value="SUBEX")
        stock_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.stock_var,
            values=list(STOCK_SYMBOLS.keys()),
            width=150,
            fg_color=COLORS['primary']
        )
        stock_dropdown.grid(row=0, column=1, padx=5, pady=15)
        
        # Period selector
        ctk.CTkLabel(controls_frame, text="Period:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=2, padx=(20, 5), pady=15
        )
        
        self.period_var = ctk.StringVar(value="2y")
        period_dropdown = ctk.CTkOptionMenu(
            controls_frame,
            variable=self.period_var,
            values=["1y", "2y", "5y", "max"],
            width=100
        )
        period_dropdown.grid(row=0, column=3, padx=5, pady=15, sticky="w")
        
        # Prediction days
        ctk.CTkLabel(controls_frame, text="Predict Days:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=4, padx=(20, 5), pady=15
        )
        
        self.predict_days_var = ctk.StringVar(value="30")
        days_entry = ctk.CTkEntry(controls_frame, textvariable=self.predict_days_var, width=60)
        days_entry.grid(row=0, column=5, padx=5, pady=15)
        
        # Action buttons
        fetch_btn = ctk.CTkButton(
            controls_frame,
            text="📥 Fetch Data",
            command=self._fetch_stock_data,
            fg_color=COLORS['secondary'],
            hover_color="#2E7D32",
            width=120
        )
        fetch_btn.grid(row=0, column=6, padx=10, pady=15)
        
        train_btn = ctk.CTkButton(
            controls_frame,
            text="🧠 Train & Predict",
            command=self._train_stock_model,
            fg_color=COLORS['primary'],
            width=140
        )
        train_btn.grid(row=0, column=7, padx=10, pady=15)
        
        # Results area
        results_frame = ctk.CTkFrame(self.tab_stock, fg_color=COLORS['card'])
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        results_frame.grid_columnconfigure(0, weight=3)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        # Chart area
        self.stock_chart_frame = ctk.CTkFrame(results_frame, fg_color=COLORS['background'])
        self.stock_chart_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Stats panel
        self.stock_stats_frame = ctk.CTkFrame(results_frame, fg_color=COLORS['card'])
        self.stock_stats_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        
        self._create_stock_placeholder()
        
    def _create_stock_placeholder(self):
        """Create placeholder for stock chart."""
        placeholder = ctk.CTkLabel(
            self.stock_chart_frame,
            text="📈 Select a stock and click 'Fetch Data' to begin\n\n"
                 "Supported stocks: SUBEX, TCS, INFOSYS, WIPRO, RELIANCE, HDFC, ICICI",
            font=ctk.CTkFont(size=14),
            text_color="#666666"
        )
        placeholder.pack(expand=True)
        
    def _build_sales_tab(self):
        """Build Sales Forecast tab content."""
        # Controls Frame
        controls_frame = ctk.CTkFrame(self.tab_sales, fg_color=COLORS['card'])
        controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        controls_frame.grid_columnconfigure(2, weight=1)
        
        # Sample size
        ctk.CTkLabel(controls_frame, text="Sample Size:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=(15, 5), pady=15
        )
        
        self.sample_size_var = ctk.StringVar(value="5000")
        sample_entry = ctk.CTkEntry(controls_frame, textvariable=self.sample_size_var, width=80)
        sample_entry.grid(row=0, column=1, padx=5, pady=15)
        
        # Buttons
        generate_btn = ctk.CTkButton(
            controls_frame,
            text="📦 Generate Data",
            command=self._generate_sales_data,
            fg_color=COLORS['secondary'],
            width=140
        )
        generate_btn.grid(row=0, column=3, padx=10, pady=15)
        
        train_btn = ctk.CTkButton(
            controls_frame,
            text="🧠 Train Model",
            command=self._train_sales_model,
            fg_color=COLORS['primary'],
            width=130
        )
        train_btn.grid(row=0, column=4, padx=10, pady=15)
        
        forecast_btn = ctk.CTkButton(
            controls_frame,
            text="📊 View Forecast",
            command=self._show_sales_forecast,
            fg_color=COLORS['warning'],
            width=130
        )
        forecast_btn.grid(row=0, column=5, padx=10, pady=15)
        
        # Results area
        results_frame = ctk.CTkFrame(self.tab_sales, fg_color=COLORS['card'])
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        results_frame.grid_columnconfigure(0, weight=2)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        # Chart area
        self.sales_chart_frame = ctk.CTkFrame(results_frame, fg_color=COLORS['background'])
        self.sales_chart_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Stats panel
        self.sales_stats_frame = ctk.CTkFrame(results_frame, fg_color=COLORS['card'])
        self.sales_stats_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        
        self._create_sales_placeholder()
        
    def _create_sales_placeholder(self):
        """Create placeholder for sales chart."""
        placeholder = ctk.CTkLabel(
            self.sales_chart_frame,
            text="🛒 Click 'Generate Data' to create BigMart-style retail dataset\n\n"
                 "The model will predict item-level sales for inventory optimization",
            font=ctk.CTkFont(size=14),
            text_color="#666666"
        )
        placeholder.pack(expand=True)
        
    def _build_churn_tab(self):
        """Build Churn Analyzer tab content."""
        # Controls Frame
        controls_frame = ctk.CTkFrame(self.tab_churn, fg_color=COLORS['card'])
        controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        controls_frame.grid_columnconfigure(2, weight=1)
        
        # Customer count
        ctk.CTkLabel(controls_frame, text="Customers:", font=ctk.CTkFont(size=12)).grid(
            row=0, column=0, padx=(15, 5), pady=15
        )
        
        self.customer_count_var = ctk.StringVar(value="3000")
        customer_entry = ctk.CTkEntry(controls_frame, textvariable=self.customer_count_var, width=80)
        customer_entry.grid(row=0, column=1, padx=5, pady=15)
        
        # Buttons
        generate_btn = ctk.CTkButton(
            controls_frame,
            text="👥 Generate Data",
            command=self._generate_churn_data,
            fg_color=COLORS['secondary'],
            width=140
        )
        generate_btn.grid(row=0, column=3, padx=10, pady=15)
        
        train_btn = ctk.CTkButton(
            controls_frame,
            text="🧠 Train Model",
            command=self._train_churn_model,
            fg_color=COLORS['primary'],
            width=130
        )
        train_btn.grid(row=0, column=4, padx=10, pady=15)
        
        analyze_btn = ctk.CTkButton(
            controls_frame,
            text="⚠️ Analyze Risk",
            command=self._analyze_churn_risk,
            fg_color=COLORS['danger'],
            width=130
        )
        analyze_btn.grid(row=0, column=5, padx=10, pady=15)
        
        # Results area
        results_frame = ctk.CTkFrame(self.tab_churn, fg_color=COLORS['card'])
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        # Left panel - Risk distribution
        self.churn_chart_frame = ctk.CTkFrame(results_frame, fg_color=COLORS['background'])
        self.churn_chart_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Right panel - High risk customers
        self.churn_list_frame = ctk.CTkFrame(results_frame, fg_color=COLORS['card'])
        self.churn_list_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        
        self._create_churn_placeholder()
        
    def _create_churn_placeholder(self):
        """Create placeholder for churn chart."""
        placeholder = ctk.CTkLabel(
            self.churn_chart_frame,
            text="👥 Click 'Generate Data' to create fintech customer dataset\n\n"
                 "The model will identify customers at risk of churning",
            font=ctk.CTkFont(size=14),
            text_color="#666666"
        )
        placeholder.pack(expand=True)
        
    def _create_status_bar(self):
        """Create status bar at bottom."""
        self.status_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLORS['card'], height=30)
        self.status_frame.grid(row=2, column=0, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.status_label.grid(row=0, column=0, padx=15, pady=5, sticky="w")
        
    def _update_status(self, message: str):
        """Update status bar message."""
        self.status_label.configure(text=message)
        self.update()
        
    def _clear_frame(self, frame):
        """Clear all widgets from a frame."""
        for widget in frame.winfo_children():
            widget.destroy()
            
    def _embed_chart(self, fig: Figure, parent_frame) -> FigureCanvasTkAgg:
        """Embed matplotlib figure in tkinter frame."""
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        self.current_canvases.append(canvas)
        return canvas
        
    # ========== STOCK PREDICTOR METHODS ==========
    
    def _fetch_stock_data(self):
        """Fetch stock data from Yahoo Finance."""
        self._update_status(f"Fetching data for {self.stock_var.get()}...")
        
        try:
            self.stock_predictor = StockPredictor(self.stock_var.get())
            data = self.stock_predictor.fetch_data(self.period_var.get())
            
            self._clear_frame(self.stock_chart_frame)
            self._clear_frame(self.stock_stats_frame)
            
            # Display chart
            setup_dark_style()
            fig = create_stock_chart(
                data.index, data['Close'].values,
                title=f"{self.stock_var.get()} - Historical Prices"
            )
            self._embed_chart(fig, self.stock_chart_frame)
            
            # Display stats
            self._show_stock_stats(data)
            
            self._update_status(f"Loaded {len(data)} days of data for {self.stock_var.get()}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch data: {str(e)}")
            self._update_status("Error fetching data")
            
    def _show_stock_stats(self, data: pd.DataFrame):
        """Display stock statistics."""
        stats = [
            ("Current Price", f"₹{data['Close'].iloc[-1]:.2f}"),
            ("Day High", f"₹{data['High'].iloc[-1]:.2f}"),
            ("Day Low", f"₹{data['Low'].iloc[-1]:.2f}"),
            ("52-Week High", f"₹{data['High'].max():.2f}"),
            ("52-Week Low", f"₹{data['Low'].min():.2f}"),
            ("Avg Volume", f"{data['Volume'].mean():,.0f}"),
            ("Data Points", f"{len(data)}"),
        ]
        
        title = ctk.CTkLabel(
            self.stock_stats_frame,
            text="📊 Stock Stats",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['primary']
        )
        title.pack(pady=(15, 10))
        
        for label, value in stats:
            frame = ctk.CTkFrame(self.stock_stats_frame, fg_color="transparent")
            frame.pack(fill="x", padx=15, pady=3)
            
            ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
            ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            
    def _train_stock_model(self):
        """Train stock prediction model."""
        if self.stock_predictor is None:
            messagebox.showwarning("Warning", "Please fetch stock data first")
            return
            
        self._update_status("Training LSTM model... This may take a few minutes")
        
        def train_thread():
            try:
                # Prepare data
                X_train, X_test, y_train, y_test = self.stock_predictor.prepare_data()
                
                # Check if TensorFlow is available
                try:
                    self.stock_predictor.build_model()
                    self.stock_predictor.train(X_train, y_train, X_test, y_test)
                    
                    # Evaluate
                    metrics = self.stock_predictor.evaluate(X_test, y_test)
                    
                    # Predict future
                    days = int(self.predict_days_var.get())
                    future_dates, predictions = self.stock_predictor.predict_future(days)
                    
                    # Update UI in main thread
                    self.after(0, lambda: self._display_stock_predictions(
                        predictions, future_dates, metrics
                    ))
                    
                except ImportError:
                    self.after(0, lambda: messagebox.showwarning(
                        "TensorFlow Required",
                        "LSTM requires TensorFlow. Please install: pip install tensorflow"
                    ))
                    
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
                
        threading.Thread(target=train_thread, daemon=True).start()
        
    def _display_stock_predictions(self, predictions, future_dates, metrics):
        """Display stock predictions."""
        self._clear_frame(self.stock_chart_frame)
        self._clear_frame(self.stock_stats_frame)
        
        # Create combined chart
        setup_dark_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical data
        data = self.stock_predictor.data
        ax.plot(data.index[-100:], data['Close'].iloc[-100:], 
                label='Historical', color='#1E88E5', linewidth=2)
        
        # Predictions
        ax.plot(future_dates, predictions, 
                label='Predicted', color='#43A047', linewidth=2, linestyle='--')
        
        ax.set_title(f"{self.stock_var.get()} - Price Prediction", fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        self._embed_chart(fig, self.stock_chart_frame)
        
        # Show metrics
        stats = [
            ("Model Metrics", ""),
            ("RMSE", f"₹{metrics['rmse']:.2f}"),
            ("MAE", f"₹{metrics['mae']:.2f}"),
            ("MAPE", f"{metrics['mape']:.2f}%"),
            ("", ""),
            ("Predictions", ""),
            ("Start Price", f"₹{predictions[0]:.2f}"),
            ("End Price", f"₹{predictions[-1]:.2f}"),
            ("Predicted Change", f"{((predictions[-1]/predictions[0])-1)*100:.1f}%"),
        ]
        
        for label, value in stats:
            if label and not value:
                lbl = ctk.CTkLabel(
                    self.stock_stats_frame, text=label,
                    font=ctk.CTkFont(size=13, weight="bold"),
                    text_color=COLORS['primary']
                )
                lbl.pack(pady=(15, 5))
            elif label:
                frame = ctk.CTkFrame(self.stock_stats_frame, fg_color="transparent")
                frame.pack(fill="x", padx=15, pady=2)
                ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
                ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
                
        self._update_status(f"Model trained. RMSE: ₹{metrics['rmse']:.2f}")
        
    # ========== SALES FORECASTER METHODS ==========
    
    def _generate_sales_data(self):
        """Generate sample sales data."""
        self._update_status("Generating retail dataset...")
        
        try:
            self.sales_forecaster = SalesForecaster()
            n_samples = int(self.sample_size_var.get())
            data = self.sales_forecaster.generate_sample_data(n_samples)
            
            self._clear_frame(self.sales_chart_frame)
            self._clear_frame(self.sales_stats_frame)
            
            # Show data distribution
            setup_dark_style()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Sales by outlet type
            outlet_sales = data.groupby('Outlet_Type')['Item_Outlet_Sales'].mean()
            axes[0].bar(range(len(outlet_sales)), outlet_sales.values, color='#1E88E5')
            axes[0].set_xticks(range(len(outlet_sales)))
            axes[0].set_xticklabels([t[:15] for t in outlet_sales.index], rotation=45, ha='right')
            axes[0].set_title('Avg Sales by Outlet Type', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Avg Sales (₹)')
            
            # Sales distribution
            axes[1].hist(data['Item_Outlet_Sales'], bins=50, color='#43A047', alpha=0.7)
            axes[1].set_title('Sales Distribution', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Sales (₹)')
            axes[1].set_ylabel('Frequency')
            
            plt.tight_layout()
            self._embed_chart(fig, self.sales_chart_frame)
            
            # Show stats
            self._show_sales_stats(data)
            
            self._update_status(f"Generated {n_samples} retail records")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def _show_sales_stats(self, data: pd.DataFrame):
        """Display sales data statistics."""
        stats = [
            ("Total Records", f"{len(data):,}"),
            ("Outlets", f"{data['Outlet_Identifier'].nunique()}"),
            ("Item Types", f"{data['Item_Type'].nunique()}"),
            ("Avg Sales", f"₹{data['Item_Outlet_Sales'].mean():,.0f}"),
            ("Max Sales", f"₹{data['Item_Outlet_Sales'].max():,.0f}"),
            ("Total Revenue", f"₹{data['Item_Outlet_Sales'].sum():,.0f}"),
        ]
        
        title = ctk.CTkLabel(
            self.sales_stats_frame,
            text="📦 Dataset Info",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['secondary']
        )
        title.pack(pady=(15, 10))
        
        for label, value in stats:
            frame = ctk.CTkFrame(self.sales_stats_frame, fg_color="transparent")
            frame.pack(fill="x", padx=15, pady=3)
            ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
            ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            
    def _train_sales_model(self):
        """Train sales forecasting model."""
        if self.sales_forecaster is None:
            messagebox.showwarning("Warning", "Please generate sales data first")
            return
            
        self._update_status("Training Gradient Boosting model...")
        
        try:
            metrics = self.sales_forecaster.train()
            
            self._clear_frame(self.sales_chart_frame)
            
            # Show feature importance
            importance = self.sales_forecaster.get_feature_importance()
            
            setup_dark_style()
            fig = create_feature_importance_chart(
                importance['Feature'].tolist(),
                importance['Importance'].values,
                title="Feature Importance - Sales Model"
            )
            self._embed_chart(fig, self.sales_chart_frame)
            
            # Update stats
            self._clear_frame(self.sales_stats_frame)
            
            stats = [
                ("Model Performance", ""),
                ("Train RMSE", f"₹{metrics['train_rmse']:.2f}"),
                ("Test RMSE", f"₹{metrics['test_rmse']:.2f}"),
                ("Train R²", f"{metrics['train_r2']:.4f}"),
                ("Test R²", f"{metrics['test_r2']:.4f}"),
            ]
            
            for label, value in stats:
                if not value:
                    lbl = ctk.CTkLabel(
                        self.sales_stats_frame, text=label,
                        font=ctk.CTkFont(size=13, weight="bold"),
                        text_color=COLORS['secondary']
                    )
                    lbl.pack(pady=(15, 5))
                else:
                    frame = ctk.CTkFrame(self.sales_stats_frame, fg_color="transparent")
                    frame.pack(fill="x", padx=15, pady=3)
                    ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
                    ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
                    
            self._update_status(f"Model trained. Test R²: {metrics['test_r2']:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def _show_sales_forecast(self):
        """Show category-wise sales forecast."""
        if self.sales_forecaster is None or self.sales_forecaster.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
            
        try:
            forecast = self.sales_forecaster.forecast_by_category()
            
            self._clear_frame(self.sales_chart_frame)
            
            setup_dark_style()
            fig = create_forecast_chart(
                forecast['Item_Category'].tolist(),
                forecast['Total_Actual'].values,
                forecast['Total_Predicted'].values,
                title="Category-wise Sales Forecast"
            )
            self._embed_chart(fig, self.sales_chart_frame)
            
            self._update_status("Showing category-wise forecast")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    # ========== CHURN ANALYZER METHODS ==========
    
    def _generate_churn_data(self):
        """Generate sample customer data."""
        self._update_status("Generating fintech customer data...")
        
        try:
            self.churn_analyzer = ChurnAnalyzer()
            n_customers = int(self.customer_count_var.get())
            data = self.churn_analyzer.generate_sample_data(n_customers)
            
            self._clear_frame(self.churn_chart_frame)
            self._clear_frame(self.churn_list_frame)
            
            # Show churn distribution
            setup_dark_style()
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Churn by tenure
            tenure_churn = data.groupby(
                pd.cut(data['tenure_months'], bins=[0, 6, 12, 24, 48, 999])
            )['churned'].mean() * 100
            axes[0].bar(range(len(tenure_churn)), tenure_churn.values, color='#E53935')
            axes[0].set_xticks(range(len(tenure_churn)))
            axes[0].set_xticklabels(['0-6m', '6-12m', '1-2y', '2-4y', '4y+'], rotation=45)
            axes[0].set_title('Churn Rate by Tenure', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Churn Rate (%)')
            
            # Churn distribution
            churned_counts = data['churned'].value_counts()
            colors = ['#43A047', '#E53935']
            axes[1].pie(churned_counts.values, labels=['Active', 'Churned'],
                       colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Customer Status', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            self._embed_chart(fig, self.churn_chart_frame)
            
            # Show stats
            self._show_churn_stats(data)
            
            self._update_status(f"Generated {n_customers} customer records")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def _show_churn_stats(self, data: pd.DataFrame):
        """Display churn data statistics."""
        churn_rate = data['churned'].mean() * 100
        
        stats = [
            ("Total Customers", f"{len(data):,}"),
            ("Churned", f"{data['churned'].sum():,}"),
            ("Active", f"{len(data) - data['churned'].sum():,}"),
            ("Churn Rate", f"{churn_rate:.1f}%"),
            ("Avg Tenure", f"{data['tenure_months'].mean():.1f} months"),
            ("Avg Balance", f"₹{data['current_balance'].mean():,.0f}"),
        ]
        
        title = ctk.CTkLabel(
            self.churn_list_frame,
            text="👥 Dataset Info",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['danger']
        )
        title.pack(pady=(15, 10))
        
        for label, value in stats:
            frame = ctk.CTkFrame(self.churn_list_frame, fg_color="transparent")
            frame.pack(fill="x", padx=15, pady=3)
            ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
            ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            
    def _train_churn_model(self):
        """Train churn prediction model."""
        if self.churn_analyzer is None:
            messagebox.showwarning("Warning", "Please generate customer data first")
            return
            
        self._update_status("Training Logistic Regression model...")
        
        try:
            metrics = self.churn_analyzer.train()
            
            self._clear_frame(self.churn_chart_frame)
            
            # Show feature importance
            importance = self.churn_analyzer.get_feature_importance()
            
            setup_dark_style()
            fig, ax = plt.subplots(figsize=(10, 6))
            
            top_features = importance.head(10)
            colors = ['#E53935' if c > 0 else '#43A047' for c in top_features['Coefficient']]
            
            ax.barh(range(len(top_features)), top_features['Abs_Importance'].values, color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'].values)
            ax.set_xlabel('Absolute Coefficient')
            ax.set_title('Feature Impact on Churn (Red = Increases, Green = Decreases)',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            self._embed_chart(fig, self.churn_chart_frame)
            
            # Update stats
            self._clear_frame(self.churn_list_frame)
            
            stats = [
                ("Model Performance", ""),
                ("Accuracy", f"{metrics['test_accuracy']*100:.1f}%"),
                ("Precision", f"{metrics['precision']*100:.1f}%"),
                ("Recall", f"{metrics['recall']*100:.1f}%"),
                ("F1 Score", f"{metrics['f1_score']*100:.1f}%"),
                ("ROC-AUC", f"{metrics['roc_auc']:.4f}"),
            ]
            
            for label, value in stats:
                if not value:
                    lbl = ctk.CTkLabel(
                        self.churn_list_frame, text=label,
                        font=ctk.CTkFont(size=13, weight="bold"),
                        text_color=COLORS['danger']
                    )
                    lbl.pack(pady=(15, 5))
                else:
                    frame = ctk.CTkFrame(self.churn_list_frame, fg_color="transparent")
                    frame.pack(fill="x", padx=15, pady=3)
                    ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=11), text_color="#888888").pack(anchor="w")
                    ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w")
                    
            self._update_status(f"Model trained. Accuracy: {metrics['test_accuracy']*100:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def _analyze_churn_risk(self):
        """Analyze and display high-risk customers."""
        if self.churn_analyzer is None or self.churn_analyzer.model is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
            
        try:
            summary = self.churn_analyzer.get_churn_summary()
            high_risk = self.churn_analyzer.get_high_risk_customers(self.churn_analyzer.data)
            
            self._clear_frame(self.churn_chart_frame)
            
            # Create risk distribution chart
            setup_dark_style()
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            # Pie chart of risk levels
            risk_counts = [summary['high_risk_count'], summary['medium_risk_count'], summary['low_risk_count']]
            colors = ['#E53935', '#FB8C00', '#43A047']
            axes[0].pie(risk_counts, labels=['High', 'Medium', 'Low'],
                       colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Risk Distribution', fontsize=12, fontweight='bold')
            
            # High-risk trend
            if len(high_risk) > 0:
                axes[1].hist(high_risk['churn_probability'], bins=20, color='#E53935', alpha=0.7)
                axes[1].axvline(x=0.7, color='white', linestyle='--', label='High Risk Threshold')
                axes[1].set_title('High Risk Probability Distribution', fontsize=12, fontweight='bold')
                axes[1].set_xlabel('Churn Probability')
                axes[1].set_ylabel('Count')
                axes[1].legend()
            
            plt.tight_layout()
            self._embed_chart(fig, self.churn_chart_frame)
            
            # Show high-risk customers
            self._clear_frame(self.churn_list_frame)
            
            title = ctk.CTkLabel(
                self.churn_list_frame,
                text="⚠️ High Risk Customers",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=COLORS['danger']
            )
            title.pack(pady=(15, 10))
            
            # Summary stats
            stats_frame = ctk.CTkFrame(self.churn_list_frame, fg_color=COLORS['background'])
            stats_frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(
                stats_frame,
                text=f"High Risk: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=COLORS['danger']
            ).pack(pady=5)
            
            ctk.CTkLabel(
                stats_frame,
                text=f"Revenue at Risk: ₹{summary['estimated_revenue_at_risk']:,.0f}",
                font=ctk.CTkFont(size=11),
                text_color="#888888"
            ).pack(pady=2)
            
            # Customer list (scrollable)
            list_title = ctk.CTkLabel(
                self.churn_list_frame,
                text="Top 10 At-Risk Customers:",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            list_title.pack(pady=(15, 5))
            
            for i, (_, customer) in enumerate(high_risk.head(10).iterrows()):
                cust_frame = ctk.CTkFrame(self.churn_list_frame, fg_color=COLORS['background'])
                cust_frame.pack(fill="x", padx=10, pady=2)
                
                ctk.CTkLabel(
                    cust_frame,
                    text=f"{customer.get('customer_id', f'Customer {i+1}')}",
                    font=ctk.CTkFont(size=11, weight="bold")
                ).pack(side="left", padx=10, pady=5)
                
                ctk.CTkLabel(
                    cust_frame,
                    text=f"{customer['churn_percentage']:.1f}%",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS['danger']
                ).pack(side="right", padx=10, pady=5)
                
            self._update_status(f"Identified {summary['high_risk_count']} high-risk customers")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    """Run the ML Analytics Suite application."""
    app = MLAnalyticsSuite()
    app.mainloop()


if __name__ == '__main__':
    main()
