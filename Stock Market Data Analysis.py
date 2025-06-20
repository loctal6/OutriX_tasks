import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class StockAnalyzer:
    
    def __init__(self, symbol, period='1y'):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.fetch_data()

    def fetch_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            print(f"Successfully fetched {len(self.data)} days of data for {self.symbol}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def calculate_moving_averages(self, short_window=20, long_window=50):
        if self.data is None:
            return
        self.data[f'MA_{short_window}'] = self.data['Close'].rolling(window=short_window).mean()
        self.data[f'MA_{long_window}'] = self.data['Close'].rolling(window=long_window).mean()
        self.data['MA_Signal'] = np.where(self.data[f'MA_{short_window}'] > self.data[f'MA_{long_window}'], 1, 0)
        self.data['MA_Position'] = self.data['MA_Signal'].diff()

    def calculate_technical_indicators(self):
        if self.data is None:
            return
        # rsi (relative strength index)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # bollinger bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)

        # volume moving average
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()

        # daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()

        # volatility (20 day rolling standard deviation)
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

    def analyze_volume_patterns(self):
        if self.data is None:
            return None, None
        # volume analysis
        volume_stats = {
            'avg_volume': self.data['Volume'].mean(),
            'max_volume': self.data['Volume'].max(),
            'min_volume': self.data['Volume'].min(),
            'volume_std': self.data['Volume'].std()
        }

        # price volume correlation
        price_volume_corr = self.data['Close'].corr(self.data['Volume'])

        # high volume days
        high_volume_threshold = volume_stats['avg_volume'] * 1.5
        self.data['High_Volume'] = self.data['Volume'] > high_volume_threshold

        return volume_stats, price_volume_corr

    def generate_market_insights(self):
        if self.data is None:
            return None

        # calculate statistics
        current_price = self.data['Close'].iloc[-1]
        price_change = ((current_price - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0]) * 100
        avg_volume = self.data['Volume'].mean()
        current_rsi = self.data['RSI'].iloc[-1]
        current_volatility = self.data['Volatility'].iloc[-1]

        # moving average signals
        ma_20_current = self.data['MA_20'].iloc[-1]
        ma_50_current = self.data['MA_50'].iloc[-1]

        insights = {
            'symbol': self.symbol,
            'current_price': round(current_price, 2),
            'period_return': round(price_change, 2),
            'average_volume': int(avg_volume),
            'current_rsi': round(current_rsi, 2),
            'current_volatility': round(current_volatility * 100, 2),
            'ma_20': round(ma_20_current, 2),
            'ma_50': round(ma_50_current, 2),
            'trend_signal': 'Bullish' if ma_20_current > ma_50_current else 'Bearish',
            'rsi_signal': 'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'
        }

        return insights

    def print_analysis_summary(self):
        insights = self.generate_market_insights()
        volume_stats, price_volume_corr = self.analyze_volume_patterns()
        if insights is None or volume_stats is None:
            print("Error: Unable to generate analysis summary")
            return

        print("=" * 60)
        print(f"STOCK MARKET ANALYSIS SUMMARY - {insights['symbol']}")
        print("=" * 60)
        print(f"Analysis Period: {self.period}")
        print(f"Data Points: {len(self.data)} trading days")
        print()

        print("PRICE ANALYSIS:")
        print(f"  Current Price: ${insights['current_price']}")
        print(f"  Period Return: {insights['period_return']}%")
        print(f"  20-Day MA: ${insights['ma_20']}")
        print(f"  50-Day MA: ${insights['ma_50']}")
        print(f"  Trend Signal: {insights['trend_signal']}")
        print()

        print("TECHNICAL INDICATORS:")
        print(f"  RSI: {insights['current_rsi']} ({insights['rsi_signal']})")
        print(f"  Volatility: {insights['current_volatility']}% (annualized)")
        print()

        print("VOLUME ANALYSIS:")
        print(f"  Average Volume: {insights['average_volume']:,}")
        print(f"  Max Volume: {int(volume_stats['max_volume']):,}")
        print(f"  Min Volume: {int(volume_stats['min_volume']):,}")
        print(f"  Price-Volume Correlation: {price_volume_corr:.3f}")
        print()

        print("MARKET BEHAVIOR INSIGHTS:")
        if insights['trend_signal'] == 'Bullish':
            print("  • Short-term momentum is positive (MA20 > MA50)")
        else:
            print("  • Short-term momentum is negative (MA20 < MA50)")

        if insights['current_volatility'] > 25:
            print("  • High volatility indicates increased market uncertainty")
        elif insights['current_volatility'] < 15:
            print("  • Low volatility suggests stable market conditions")
        else:
            print("  • Moderate volatility indicates normal market conditions")

        if abs(price_volume_corr) > 0.3:
            print(f"  • Strong price-volume relationship (correlation: {price_volume_corr:.3f})")
        else:
            print(f"  • Weak price-volume relationship (correlation: {price_volume_corr:.3f})")

        print("=" * 60)

    def create_alternative_visualizations(self):
        """Create alternative visualizations that work reliably across different environments"""
        if self.data is None:
            return

        print(f"\n Creating text-based visualizations for {self.symbol}:")

        # price trend analysis
        recent_prices = self.data['Close'].tail(10)
        print(f"\n Recent 10-day price trend:")
        for date, price in recent_prices.items():
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

        # moving average signals
        print(f"\n Moving Average Signals (Last 5 days):")
        ma_signals = self.data[['Close', 'MA_20', 'MA_50']].tail(5)
        for date, row in ma_signals.iterrows():
            signal = " BULLISH" if row['MA_20'] > row['MA_50'] else "🔴 BEARISH"
            print(
                f"{date.strftime('%Y-%m-%d')}: {signal} | Close: ${row['Close']:.2f} | MA20: ${row['MA_20']:.2f} | MA50: ${row['MA_50']:.2f}")

        # volume analysis
        print(f"\n Volume Analysis:")
        high_volume_days = self.data[self.data['High_Volume'] == True].tail(5)
        if not high_volume_days.empty:
            print("Recent high volume days:")
            for date, row in high_volume_days.iterrows():
                print(f"{date.strftime('%Y-%m-%d')}: {row['Volume']:,} shares (${row['Close']:.2f})")

        # rsi levels
        print(f"\n RSI Trend (Last 5 days):")
        rsi_data = self.data[['Close', 'RSI']].tail(5)
        for date, row in rsi_data.iterrows():
            if row['RSI'] > 70:
                signal = "OVERBOUGHT"
            elif row['RSI'] < 30:
                signal = "OVERSOLD"
            else:
                signal = "NEUTRAL"
            print(f"{date.strftime('%Y-%m-%d')}: RSI {row['RSI']:.1f} {signal}")

    def plot_matplotlib_analysis(self, save_plots=True):
        if self.data is None:
            return
        try:
            import matplotlib
            if save_plots:
                matplotlib.use('Agg')
            plt.style.use('default')
            sns.set_palette("husl")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{self.symbol} Stock Market Analysis', fontsize=16, fontweight='bold')
            axes[0, 0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2, color='black')
            axes[0, 0].plot(self.data.index, self.data['MA_20'], label='MA 20', linewidth=2, color='orange')
            axes[0, 0].plot(self.data.index, self.data['MA_50'], label='MA 50', linewidth=2, color='blue')
            axes[0, 0].set_title('Stock Price and Moving Averages')
            axes[0, 0].set_ylabel('Price ($)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 1].bar(self.data.index, self.data['Volume'], alpha=0.7, color='lightblue', label='Volume')
            axes[0, 1].plot(self.data.index, self.data['Volume_MA'], color='red', linewidth=2, label='Volume MA')
            axes[0, 1].set_title('Volume Patterns')
            axes[0, 1].set_ylabel('Volume')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].hist(self.data['Daily_Return'].dropna(), bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(self.data['Daily_Return'].mean(), color='red', linestyle='--', label='Mean')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 1].scatter(self.data['Volume'], self.data['Close'], alpha=0.5, color='purple')
            axes[1, 1].set_title('Price vs Volume Correlation')
            axes[1, 1].set_xlabel('Volume')
            axes[1, 1].set_ylabel('Close Price ($)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.tight_layout()

            if save_plots:
                filename = f'{self.symbol}_analysis.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Chart saved as {filename}")
            else:
                try:
                    plt.show()
                except Exception as e:
                    print(f"Display error: {e}")
                    filename = f'{self.symbol}_analysis.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Chart saved as {filename}")

            plt.close()

        except Exception as e:
            print(f"Error creating matplotlib plots: {e}")
            print("Continuing with analysis...")

    def plot_comprehensive_analysis(self):
        if self.data is None:
            return

        try:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'{self.symbol} Stock Price and Moving Averages',
                    'Volume Analysis',
                    'RSI (Relative Strength Index)',
                    'Bollinger Bands'
                ),
                row_heights=[0.4, 0.2, 0.2, 0.2]
            )
            fig.add_trace(go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['MA_20'],
                name='MA 20',
                line=dict(color='orange', width=2)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['MA_50'],
                name='MA 50',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            colors = ['red' if row['Close'] < row['Open'] else 'green' for index, row in self.data.iterrows()]
            fig.add_trace(go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['Volume_MA'],
                name='Volume MA',
                line=dict(color='purple', width=2)
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['RSI'],
                name='RSI',
                line=dict(color='orange')
            ), row=3, col=1)

            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['Close'],
                name='Close Price',
                line=dict(color='black')
            ), row=4, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['BB_Upper'],
                name='BB Upper',
                line=dict(color='red', dash='dash')
            ), row=4, col=1)

            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data['BB_Lower'],
                name='BB Lower',
                line=dict(color='red', dash='dash'),
                fill='tonexty'
            ), row=4, col=1)

            fig.update_layout(
                title=f'{self.symbol} Comprehensive Stock Analysis',
                xaxis_title='Date',
                height=1000,
                showlegend=True
            )

            fig.show()

        except Exception as e:
            print(f"Error creating interactive plots: {e}")


# Example usage and demonstration
def main():

    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']

    print("COMPREHENSIVE STOCK MARKET DATA ANALYSIS")
    print("=" * 50)
    print("This project analyzes time-series stock data and visualizes:")
    print("• Price trends and patterns")
    print("• Moving averages (short-term and long-term)")
    print("• Volume patterns and correlations")
    print("• Technical indicators (RSI, Bollinger Bands)")
    print("• Market behavior insights")

    for symbol in stocks:
        print(f"\nAnalyzing {symbol}...")

        try:
            analyzer = StockAnalyzer(symbol, period='1y')

            if analyzer.data is not None:
                analyzer.calculate_moving_averages(20, 50)
                analyzer.calculate_technical_indicators()
                analyzer.print_analysis_summary()
                analyzer.create_alternative_visualizations()
                print(f"\nGenerating chart visualizations for {symbol}...")
                analyzer.plot_matplotlib_analysis(save_plots=True)

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

        print("-" * 50)

    print("\nAnalysis complete! The project demonstrates:")
    print("-Data wrangling with yfinance and pandas")
    print("-Time-series analysis and trend identification")
    print("-Moving average calculations and signals")
    print("-Volume pattern analysis")
    print("-Technical indicator computation")
    print("-Comprehensive visualization with matplotlib and plotly")
    print("-Market behavior explanation and insights")


if __name__ == "__main__":
    main()
