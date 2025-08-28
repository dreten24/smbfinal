## File: app.py
#!/usr/bin/env python3
"""
Scan My Biz - Professional Bank Statement Analyzer
Advanced MCA underwriting tool with PDF parsing capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import io
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.error("PDF libraries not installed. Run: pip install PyPDF2 pdfplumber")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Transaction:
    """Represents a single bank transaction"""
    date: datetime
    description: str
    amount: float
    balance: Optional[float] = None
    category: Optional[str] = None

@dataclass
class AnalysisResult:
    """Results of bank statement analysis"""
    total_deposits: float
    total_withdrawals: float
    average_balance: float
    minimum_balance: float
    negative_days: int
    transaction_count: int
    date_range: Tuple[datetime, datetime]
    volatility_score: float
    risk_grade: str
    mca_payments: List[Dict]
    daily_balances: pd.DataFrame
    transactions: pd.DataFrame

class BankStatementParser:
    """Universal bank statement parser"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.txt']
        self.debug_info = []
        
    def parse_file(self, file) -> List[Transaction]:
        """Parse uploaded file and extract transactions"""
        self.debug_info.clear()
        
        file_extension = Path(file.name).suffix.lower()
        self.debug_info.append(f"Processing file: {file.name} ({file_extension})")
        
        if file_extension == '.pdf':
            return self._parse_pdf(file)
        elif file_extension == '.csv':
            return self._parse_csv(file)
        elif file_extension == '.txt':
            return self._parse_text(file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _parse_pdf(self, file) -> List[Transaction]:
        """Parse PDF bank statement"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF parsing libraries not available")
        
        transactions = []
        text_content = ""
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file) as pdf:
                for page_num, page in enumerate(pdf.pages[:15]):  # Limit to first 15 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                        self.debug_info.append(f"Extracted text from page {page_num + 1}")
        except Exception as e:
            self.debug_info.append(f"pdfplumber failed: {e}, trying PyPDF2")
            try:
                # Fallback to PyPDF2
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(min(len(pdf_reader.pages), 15)):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    text_content += page_text + "\n"
            except Exception as e2:
                raise Exception(f"Both PDF parsers failed: pdfplumber({e}), PyPDF2({e2})")
        
        if not text_content.strip():
            raise ValueError("No text extracted from PDF")
        
        self.debug_info.append(f"Extracted {len(text_content)} characters of text")
        return self._parse_text_content(text_content)
    
    def _parse_csv(self, file) -> List[Transaction]:
        """Parse CSV bank statement"""
        try:
            df = pd.read_csv(file)
            self.debug_info.append(f"CSV loaded with {len(df)} rows and columns: {list(df.columns)}")
            
            transactions = []
            
            # Common column name mappings
            date_cols = ['date', 'Date', 'Transaction Date', 'Posted Date', 'transaction_date']
            desc_cols = ['description', 'Description', 'Memo', 'Details', 'Transaction Details']
            amount_cols = ['amount', 'Amount', 'Transaction Amount', 'Credit', 'Debit']
            balance_cols = ['balance', 'Balance', 'Running Balance', 'Account Balance']
            
            # Find the actual column names
            date_col = next((col for col in date_cols if col in df.columns), None)
            desc_col = next((col for col in desc_cols if col in df.columns), None)
            amount_col = next((col for col in amount_cols if col in df.columns), None)
            balance_col = next((col for col in balance_cols if col in df.columns), None)
            
            if not all([date_col, desc_col, amount_col]):
                raise ValueError(f"Required columns not found. Available: {list(df.columns)}")
            
            for _, row in df.iterrows():
                try:
                    date = pd.to_datetime(row[date_col])
                    description = str(row[desc_col]).strip()
                    amount = float(str(row[amount_col]).replace('$', '').replace(',', '').replace('(', '-').replace(')', ''))
                    balance = float(str(row[balance_col]).replace('$', '').replace(',', '')) if balance_col and pd.notna(row[balance_col]) else None
                    
                    transactions.append(Transaction(date=date, description=description, amount=amount, balance=balance))
                except (ValueError, TypeError) as e:
                    self.debug_info.append(f"Skipped row due to parsing error: {e}")
                    continue
            
            self.debug_info.append(f"Successfully parsed {len(transactions)} transactions from CSV")
            return transactions
            
        except Exception as e:
            raise Exception(f"CSV parsing failed: {e}")
    
    def _parse_text(self, file) -> List[Transaction]:
        """Parse text file"""
        text_content = file.read().decode('utf-8')
        return self._parse_text_content(text_content)
    
    def _parse_text_content(self, text: str) -> List[Transaction]:
        """Parse transactions from text content (universal method)"""
        transactions = []
        lines = text.split('\n')
        
        # Date patterns for different formats
        date_patterns = [
            r'^(\d{1,2}/\d{1,2}/\d{2,4})',  # MM/DD/YYYY
            r'^(\d{1,2}/\d{1,2})\s',        # MM/DD (Capital One style)
            r'^(\d{4}-\d{1,2}-\d{1,2})',    # YYYY-MM-DD
            r'^(\d{1,2}-\d{1,2}-\d{2,4})',  # MM-DD-YYYY
            r'(\d{1,2}/\d{1,2}/\d{2,4})',   # MM/DD/YYYY anywhere in line
        ]
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if len(line) < 10:
                continue
            
            # Skip headers and footers
            skip_keywords = ['account detail', 'continued for period', 'page', 'member fdic', 
                           'products and services', 'date', 'description', 'amount', 'balance']
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
            
            # Find date in line
            date_match = None
            date_str = None
            
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    date_str = match.group(1)
                    date_match = match
                    break
            
            if not date_match:
                continue
            
            # Parse date
            try:
                if '/' in date_str and len(date_str.split('/')) == 2:
                    # MM/DD format - assume current year
                    month, day = date_str.split('/')
                    date = datetime(2024, int(month), int(day))
                else:
                    date = pd.to_datetime(date_str)
            except:
                continue
            
            # Remove date from line and find dollar amounts
            remaining_text = line[date_match.end():].strip()
            dollar_amounts = re.findall(r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', remaining_text)
            
            if not dollar_amounts:
                continue
            
            # Extract description (text before first dollar sign)
            dollar_index = remaining_text.find('$')
            description = remaining_text[:dollar_index].strip() if dollar_index > 0 else remaining_text.split('$')[0].strip()
            
            if len(description) < 3:
                # Try next line for description
                if line_num + 1 < len(lines):
                    next_line = lines[line_num + 1].strip()
                    if next_line and not re.search(r'^\d', next_line) and len(next_line) > 3:
                        description = next_line
            
            if len(description) < 3:
                continue
            
            # Determine transaction amount and balance
            amounts = [float(amt.replace(',', '')) for amt in dollar_amounts]
            
            if len(amounts) == 1:
                transaction_amount = amounts[0]
                balance = None
            elif len(amounts) >= 2:
                # Usually transaction amount and ending balance
                transaction_amount = amounts[-2]  # Second to last
                balance = amounts[-1]  # Last amount is ending balance
            else:
                continue
            
            # Determine if deposit (positive) or withdrawal (negative)
            desc_lower = description.lower()
            line_lower = line.lower()
            
            # Positive indicators (deposits)
            positive_keywords = ['deposit', 'credit', 'uber', 'doordash', 'square', 'paypal',
                               'venmo', 'zelle', 'transfer in', 'refund', 'interest', 'dividend',
                               'ach deposit', 'direct deposit', 'merch bnkcd', 'fdms', 'customer deposit']
            
            # Negative indicators (withdrawals)
            negative_keywords = ['withdrawal', 'payment', 'purchase', 'check', 'fee', 'charge',
                               'debit', 'atm', 'transfer out', 'loan', 'ach withdrawal', 'wire',
                               'auto pay', 'bill pay']
            
            if any(keyword in desc_lower or keyword in line_lower for keyword in positive_keywords):
                transaction_amount = abs(transaction_amount)
            elif any(keyword in desc_lower or keyword in line_lower for keyword in negative_keywords):
                transaction_amount = -abs(transaction_amount)
            else:
                # Default assumption for unclear transactions
                transaction_amount = -abs(transaction_amount)
            
            transactions.append(Transaction(
                date=date,
                description=description[:100],  # Limit length
                amount=transaction_amount,
                balance=balance
            ))
        
        self.debug_info.append(f"Parsed {len(transactions)} transactions from text")
        return sorted(transactions, key=lambda t: t.date)

class BankStatementAnalyzer:
    """Analyzes parsed bank statement data"""
    
    def __init__(self):
        self.mca_keywords = [
            'ondeck', 'kabbage', 'bluevine', 'funding circle', 'square capital',
            'paypal working capital', 'merchant cash', 'advance', 'capify',
            'credibly', 'lendio', 'forward financing', 'rapid finance'
        ]
    
    def analyze(self, transactions: List[Transaction]) -> AnalysisResult:
        """Perform comprehensive analysis of transactions"""
        if not transactions:
            raise ValueError("No transactions to analyze")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([
            {
                'date': t.date,
                'description': t.description,
                'amount': t.amount,
                'balance': t.balance
            }
            for t in transactions
        ])
        
        # Calculate basic metrics
        deposits = df[df['amount'] > 0]['amount'].sum()
        withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
        
        # Daily balances analysis
        daily_balances = self._calculate_daily_balances(df)
        
        # Count negative days (days when ending balance < 0)
        negative_days = len(daily_balances[daily_balances['balance'] < 0])
        
        # Calculate averages and volatility
        avg_balance = daily_balances['balance'].mean()
        min_balance = daily_balances['balance'].min()
        volatility = self._calculate_volatility(daily_balances)
        
        # Risk assessment
        risk_grade = self._calculate_risk_grade(avg_balance, negative_days, len(daily_balances))
        
        # MCA detection
        mca_payments = self._detect_mca_payments(df)
        
        return AnalysisResult(
            total_deposits=deposits,
            total_withdrawals=withdrawals,
            average_balance=avg_balance,
            minimum_balance=min_balance,
            negative_days=negative_days,
            transaction_count=len(df),
            date_range=(df['date'].min(), df['date'].max()),
            volatility_score=volatility,
            risk_grade=risk_grade,
            mca_payments=mca_payments,
            daily_balances=daily_balances,
            transactions=df
        )
    
    def _calculate_daily_balances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily ending balances"""
        # If balance column exists, use those values
        if 'balance' in df.columns and df['balance'].notna().any():
            # Group by date and take the last balance for each day
            daily_balances = df.groupby(df['date'].dt.date).agg({
                'balance': 'last'
            }).reset_index()
            daily_balances.columns = ['date', 'balance']
        else:
            # Calculate running balance from transactions
            df_sorted = df.sort_values('date')
            df_sorted['running_balance'] = df_sorted['amount'].cumsum() + 10000  # Assume starting balance
            
            daily_balances = df_sorted.groupby(df_sorted['date'].dt.date).agg({
                'running_balance': 'last'
            }).reset_index()
            daily_balances.columns = ['date', 'balance']
        
        return daily_balances.sort_values('date')
    
    def _calculate_volatility(self, daily_balances: pd.DataFrame) -> float:
        """Calculate balance volatility score"""
        if len(daily_balances) < 2:
            return 0
        
        balances = daily_balances['balance']
        mean_balance = balances.mean()
        std_balance = balances.std()
        
        if mean_balance == 0:
            return 100
        
        volatility = min(100, (std_balance / abs(mean_balance)) * 100)
        return round(volatility, 1)
    
    def _calculate_risk_grade(self, avg_balance: float, negative_days: int, total_days: int) -> str:
        """Calculate risk grade based on account performance"""
        score = 85  # Start with B+ grade
        
        # Adjust based on average balance
        if avg_balance < 5000:
            score -= 25
        elif avg_balance < 15000:
            score -= 10
        elif avg_balance > 50000:
            score += 10
        
        # Adjust based on negative days
        negative_ratio = negative_days / total_days if total_days > 0 else 0
        if negative_ratio > 0.2:  # More than 20% negative days
            score -= 30
        elif negative_ratio > 0.1:  # More than 10% negative days
            score -= 15
        
        # Convert score to grade
        if score >= 95:
            return 'A+'
        elif score >= 90:
            return 'A'
        elif score >= 85:
            return 'A-'
        elif score >= 80:
            return 'B+'
        elif score >= 75:
            return 'B'
        elif score >= 70:
            return 'B-'
        elif score >= 65:
            return 'C+'
        elif score >= 60:
            return 'C'
        else:
            return 'D'
    
    def _detect_mca_payments(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential MCA payments"""
        mca_payments = []
        
        # Look for recurring payments to known MCA providers
        outgoing_payments = df[df['amount'] < -100]  # Significant outgoing payments
        
        for keyword in self.mca_keywords:
            matching_payments = outgoing_payments[
                outgoing_payments['description'].str.contains(keyword, case=False, na=False)
            ]
            
            if len(matching_payments) >= 2:  # Recurring pattern
                avg_amount = matching_payments['amount'].mean()
                frequency = self._calculate_payment_frequency(matching_payments)
                
                mca_payments.append({
                    'provider': keyword.title(),
                    'average_amount': avg_amount,
                    'frequency': frequency,
                    'count': len(matching_payments),
                    'risk_level': 'High' if abs(avg_amount) > 1000 else 'Medium'
                })
        
        return sorted(mca_payments, key=lambda x: abs(x['average_amount']), reverse=True)
    
    def _calculate_payment_frequency(self, payments: pd.DataFrame) -> str:
        """Calculate frequency of recurring payments"""
        if len(payments) < 2:
            return 'Unknown'
        
        dates = sorted(payments['date'].tolist())
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval <= 1.5:
            return 'Daily'
        elif avg_interval <= 7.5:
            return 'Weekly'
        elif avg_interval <= 15:
            return 'Bi-weekly'
        elif avg_interval <= 35:
            return 'Monthly'
        else:
            return 'Irregular'

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Scan My Biz - Bank Statement Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .debug-panel {
        background-color: #f0f2ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Scan My Biz</h1>
        <p>Professional Bank Statement Analysis & MCA Underwriting Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = []
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Bank Statements")
        
        uploaded_files = st.file_uploader(
            "Choose bank statement files",
            type=['pdf', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, CSV, or TXT bank statements"
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
            
            if st.button("Analyze Statements", type="primary", use_container_width=True):
                with st.spinner("Processing bank statements..."):
                    try:
                        parser = BankStatementParser()
                        analyzer = BankStatementAnalyzer()
                        
                        all_transactions = []
                        debug_info = []
                        
                        # Process each file
                        for file in uploaded_files:
                            transactions = parser.parse_file(file)
                            all_transactions.extend(transactions)
                            debug_info.extend(parser.debug_info)
                        
                        if not all_transactions:
                            st.error("No transactions found in uploaded files")
                            return
                        
                        # Analyze transactions
                        result = analyzer.analyze(all_transactions)
                        st.session_state.analysis_result = result
                        st.session_state.debug_info = debug_info
                        
                        st.success("Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        logger.error(f"Analysis error: {e}")
    
    # Main content
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Debug information
        with st.expander("Processing Log", expanded=False):
            st.markdown('<div class="debug-panel">', unsafe_allow_html=True)
            for info in st.session_state.debug_info:
                st.text(f"• {info}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Metrics
        st.header("Key Financial Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>${result.average_balance:,.0f}</h3>
                <p>Average Daily Balance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{result.negative_days}</h3>
                <p>Days in Negative</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>${result.total_deposits:,.0f}</h3>
                <p>Total Deposits</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{result.risk_grade}</h3>
                <p>Risk Grade</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Secondary Metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Total Withdrawals", f"${result.total_withdrawals:,.0f}")
        
        with col6:
            st.metric("Minimum Balance", f"${result.minimum_balance:,.0f}")
        
        with col7:
            st.metric("Volatility Score", f"{result.volatility_score}%")
        
        with col8:
            st.metric("Total Transactions", f"{result.transaction_count:,}")
        
        # Charts
        st.header("Balance Trend Analysis")
        
        # Daily Balance Chart
        fig = px.line(
            result.daily_balances,
            x='date',
            y='balance',
            title='Daily Account Balance',
            labels={'balance': 'Account Balance ($)', 'date': 'Date'}
        )
        fig.update_traces(line=dict(color='#667eea', width=3))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # MCA Analysis
        if result.mca_payments:
            st.header("MCA Payment Detection")
            
            mca_df = pd.DataFrame(result.mca_payments)
            st.dataframe(
                mca_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Transaction Summary
        st.header("Transaction Summary")
        
        # Show recent transactions
        recent_transactions = result.transactions.sort_values('date', ascending=False).head(20)
        
        st.dataframe(
            recent_transactions[['date', 'description', 'amount', 'balance']].round(2),
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        st.header("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to CSV
            csv_data = result.transactions.to_csv(index=False)
            st.download_button(
                label="Download Transactions CSV",
                data=csv_data,
                file_name=f"bank_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export analysis summary
            summary = {
                "analysis_date": datetime.now().isoformat(),
                "total_deposits": result.total_deposits,
                "total_withdrawals": result.total_withdrawals,
                "average_balance": result.average_balance,
                "minimum_balance": result.minimum_balance,
                "negative_days": result.negative_days,
                "risk_grade": result.risk_grade,
                "volatility_score": result.volatility_score,
                "mca_payments": result.mca_payments
            }
            
            summary_json = json.dumps(summary, indent=2, default=str)
            st.download_button(
                label="Download Analysis Summary",
                data=summary_json,
                file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    else:
        # Welcome message
        st.info("""
        Welcome to Scan My Biz!
        
        Upload your bank statements using the sidebar to get started with professional financial analysis including:
        
        - Accurate deposit and withdrawal tracking
        - Negative balance day calculation  
        - MCA payment detection
        - Risk assessment and grading
        - Cash flow volatility analysis
        - Professional reporting and exports
        
        Supported formats: PDF, CSV, TXT files from any bank or credit union.
        """)

if __name__ == "__main__":
    main()

## File: .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Streamlit
.streamlit/
