#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan My Biz - Professional Bank Statement Analyzer
Advanced MCA underwriting tool with PDF parsing capabilities
Fixed for Render deployment
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
import traceback
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import sys
import os

# PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    """Universal bank statement parser with enhanced error handling"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.txt']
        self.debug_info = []
        
    def parse_file(self, uploaded_file) -> List[Transaction]:
        """Parse uploaded file and extract transactions"""
        self.debug_info.clear()
        transactions = []
        
        try:
            file_name = uploaded_file.name
            file_extension = Path(file_name).suffix.lower()
            self.debug_info.append(f"Processing file: {file_name} ({file_extension})")
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if file_extension == '.pdf':
                transactions = self._parse_pdf(uploaded_file)
            elif file_extension == '.csv':
                transactions = self._parse_csv(uploaded_file)
            elif file_extension == '.txt':
                transactions = self._parse_text(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            self.debug_info.append(f"Extracted {len(transactions)} transactions")
            return transactions
            
        except Exception as e:
            self.debug_info.append(f"Error parsing file: {str(e)}")
            logger.error(f"File parsing error: {str(e)}")
            return []

    def _parse_pdf(self, file) -> List[Transaction]:
        """Parse PDF bank statement"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF parsing libraries not available")
            
        transactions = []
        text_content = ""
        
        try:
            # Try pdfplumber first
            file.seek(0)
            file_bytes = file.read()
            
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                            self.debug_info.append(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        self.debug_info.append(f"Error on page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            self.debug_info.append(f"pdfplumber failed: {str(e)}, trying PyPDF2")
            
            # Fallback to PyPDF2
            try:
                file.seek(0)
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                    except Exception as e:
                        self.debug_info.append(f"PyPDF2 error on page {page_num + 1}: {str(e)}")
                        continue
                        
            except Exception as e:
                self.debug_info.append(f"Both PDF parsers failed: {str(e)}")
                return []
        
        if text_content.strip():
            transactions = self._parse_text_content(text_content)
        else:
            self.debug_info.append("No text extracted from PDF")
            
        return transactions

    def _parse_csv(self, file) -> List[Transaction]:
        """Parse CSV bank statement with enhanced error handling"""
        transactions = []
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    content = file.read().decode(encoding)
                    file_obj = io.StringIO(content)
                    df = pd.read_csv(file_obj)
                    self.debug_info.append(f"Successfully decoded with {encoding}")
                    break
                except (UnicodeDecodeError, pd.errors.EmptyDataError):
                    continue
                    
            if df is None:
                raise ValueError("Could not decode CSV file with any encoding")
                
            self.debug_info.append(f"CSV shape: {df.shape}")
            self.debug_info.append(f"CSV columns: {list(df.columns)}")
            
            # Clean and standardize column names
            df.columns = df.columns.str.strip().str.lower()
            
            # Find date column
            date_cols = ['date', 'transaction date', 'posted date', 'trans date', 'effective date']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            # Find amount columns
            amount_cols = ['amount', 'debit', 'credit', 'transaction amount', 'net amount']
            debit_cols = ['debit', 'withdrawals', 'debits', 'debit amount']
            credit_cols = ['credit', 'deposits', 'credits', 'credit amount']
            
            amount_col = None
            debit_col = None
            credit_col = None
            
            for col in df.columns:
                if any(amt in col for amt in amount_cols):
                    amount_col = col
                if any(deb in col for deb in debit_cols):
                    debit_col = col
                if any(cred in col for cred in credit_cols):
                    credit_col = col
                    
            # Find description column
            desc_cols = ['description', 'memo', 'payee', 'transaction description', 'details']
            desc_col = None
            for col in desc_cols:
                if col in df.columns:
                    desc_col = col
                    break
                    
            # Find balance column
            balance_cols = ['balance', 'running balance', 'account balance', 'ending balance']
            balance_col = None
            for col in balance_cols:
                if col in df.columns:
                    balance_col = col
                    break
            
            if not date_col:
                raise ValueError("Could not find date column in CSV")
                
            self.debug_info.append(f"Using columns - Date: {date_col}, Amount: {amount_col}, Debit: {debit_col}, Credit: {credit_col}")
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    # Parse date
                    date_str = str(row[date_col]).strip()
                    if not date_str or date_str.lower() in ['nan', 'none', '']:
                        continue
                        
                    transaction_date = self._parse_date(date_str)
                    if not transaction_date:
                        continue
                    
                    # Parse amount
                    amount = 0.0
                    if amount_col and amount_col in row.index:
                        amount = self._parse_amount(str(row[amount_col]))
                    elif debit_col and credit_col:
                        debit = self._parse_amount(str(row[debit_col])) if pd.notna(row[debit_col]) else 0
                        credit = self._parse_amount(str(row[credit_col])) if pd.notna(row[credit_col]) else 0
                        amount = credit - debit
                    else:
                        # Try to find amount in any numeric column
                        for col in df.columns:
                            if pd.api.types.is_numeric_dtype(df[col]) and pd.notna(row[col]):
                                amount = float(row[col])
                                break
                    
                    # Parse description
                    description = ""
                    if desc_col and desc_col in row.index:
                        description = str(row[desc_col]).strip()
                    
                    # Parse balance
                    balance = None
                    if balance_col and balance_col in row.index:
                        balance = self._parse_amount(str(row[balance_col]))
                    
                    if abs(amount) > 0.01:  # Only include non-zero transactions
                        transactions.append(Transaction(
                            date=transaction_date,
                            description=description,
                            amount=amount,
                            balance=balance
                        ))
                        
                except Exception as e:
                    self.debug_info.append(f"Error processing row {idx}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.debug_info.append(f"CSV parsing error: {str(e)}")
            logger.error(f"CSV parsing error: {str(e)}")
            
        return transactions

    def _parse_text(self, file) -> List[Transaction]:
        """Parse text bank statement"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    content = file.read().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
                    
            if content is None:
                raise ValueError("Could not decode text file")
                
            return self._parse_text_content(content)
            
        except Exception as e:
            self.debug_info.append(f"Text parsing error: {str(e)}")
            return []

    def _parse_text_content(self, text: str) -> List[Transaction]:
        """Parse transactions from text content"""
        transactions = []
        lines = text.split('\n')
        
        # Enhanced date patterns
        date_patterns = [
            r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
            r'\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b',
            r'\b(\d{1,2}\s+\w{3}\s+\d{4})\b',
            r'\b(\w{3}\s+\d{1,2},?\s+\d{4})\b'
        ]
        
        # Enhanced amount patterns
        amount_patterns = [
            r'[\$]?([+-]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'([+-]?\d+\.\d{2})',
            r'([+-]?\d+)',
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
                
            # Find date
            transaction_date = None
            for pattern in date_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    date_str = match.group(1)
                    transaction_date = self._parse_date(date_str)
                    if transaction_date:
                        break
                if transaction_date:
                    break
                    
            if not transaction_date:
                continue
                
            # Find amounts
            amounts = []
            for pattern in amount_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    amount_str = match.group(1)
                    amount = self._parse_amount(amount_str)
                    if amount is not None:
                        amounts.append(amount)
                        
            if not amounts:
                continue
                
            # Extract description (everything except dates and amounts)
            desc_line = line
            for pattern in date_patterns:
                desc_line = re.sub(pattern, '', desc_line)
            for pattern in amount_patterns:
                desc_line = re.sub(r'[\$]?' + pattern, '', desc_line)
            description = ' '.join(desc_line.split()).strip()
            
            # Use the most significant amount (usually the largest absolute value)
            main_amount = max(amounts, key=abs) if amounts else 0
            balance = amounts[-1] if len(amounts) > 1 else None
            
            if abs(main_amount) > 0.01:
                transactions.append(Transaction(
                    date=transaction_date,
                    description=description,
                    amount=main_amount,
                    balance=balance
                ))
                
        return transactions

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support"""
        if not date_str or str(date_str).lower() in ['nan', 'none', '']:
            return None
            
        date_str = str(date_str).strip()
        
        formats = [
            '%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y',
            '%Y/%m/%d', '%Y-%m-%d',
            '%d/%m/%Y', '%d/%m/%y', '%d-%m-%Y', '%d-%m-%y',
            '%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%d %B %Y',
            '%b %d %Y', '%B %d %Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        return None

    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string and return float"""
        if not amount_str or str(amount_str).lower() in ['nan', 'none', '']:
            return None
            
        amount_str = str(amount_str).strip()
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$,\s]', '', amount_str)
        
        # Handle parentheses for negative amounts
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]
            
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None

class BankStatementAnalyzer:
    """Analyzes parsed transactions for MCA underwriting"""
    
    def __init__(self):
        self.mca_keywords = [
            'advance', 'funding', 'capital', 'cash advance', 'merchant',
            'kabbage', 'ondeck', 'fundbox', 'bluevine', 'paypal',
            'square capital', 'amazon lending', 'quickbooks capital',
            'daily pay', 'weekly pay', 'factor', 'receivables'
        ]
    
    def analyze(self, transactions: List[Transaction]) -> AnalysisResult:
        """Perform comprehensive analysis of transactions"""
        if not transactions:
            return self._empty_analysis()
            
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'date': t.date,
                'description': t.description,
                'amount': t.amount,
                'balance': t.balance
            }
            for t in transactions
        ])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate basic metrics
        deposits = df[df['amount'] > 0]['amount'].sum()
        withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
        
        # Calculate daily balances if not provided
        if df['balance'].isna().all():
            df['balance'] = df['amount'].cumsum()
            
        # Fill missing balances
        df['balance'] = df['balance'].fillna(method='ffill')
        
        # Daily balance analysis
        daily_balances = df.groupby(df['date'].dt.date).agg({
            'balance': 'last',
            'amount': 'sum'
        }).reset_index()
        
        daily_balances['date'] = pd.to_datetime(daily_balances['date'])
        
        avg_balance = daily_balances['balance'].mean()
        min_balance = daily_balances['balance'].min()
        negative_days = (daily_balances['balance'] < 0).sum()
        
        # Volatility calculation
        balance_std = daily_balances['balance'].std()
        volatility_score = (balance_std / abs(avg_balance)) * 100 if avg_balance != 0 else 100
        
        # Risk grading
        risk_grade = self._calculate_risk_grade(avg_balance, min_balance, negative_days, volatility_score)
        
        # MCA detection
        mca_payments = self._detect_mca_payments(df)
        
        # Date range
        date_range = (df['date'].min(), df['date'].max())
        
        return AnalysisResult(
            total_deposits=deposits,
            total_withdrawals=withdrawals,
            average_balance=avg_balance,
            minimum_balance=min_balance,
            negative_days=negative_days,
            transaction_count=len(transactions),
            date_range=date_range,
            volatility_score=volatility_score,
            risk_grade=risk_grade,
            mca_payments=mca_payments,
            daily_balances=daily_balances,
            transactions=df
        )
    
    def _empty_analysis(self) -> AnalysisResult:
        """Return empty analysis result"""
        return AnalysisResult(
            total_deposits=0.0,
            total_withdrawals=0.0,
            average_balance=0.0,
            minimum_balance=0.0,
            negative_days=0,
            transaction_count=0,
            date_range=(datetime.now(), datetime.now()),
            volatility_score=0.0,
            risk_grade='N/A',
            mca_payments=[],
            daily_balances=pd.DataFrame(),
            transactions=pd.DataFrame()
        )
    
    def _calculate_risk_grade(self, avg_balance: float, min_balance: float, negative_days: int, volatility: float) -> str:
        """Calculate risk grade based on account metrics"""
        score = 0
        
        # Average balance scoring
        if avg_balance >= 50000:
            score += 30
        elif avg_balance >= 25000:
            score += 25
        elif avg_balance >= 10000:
            score += 20
        elif avg_balance >= 5000:
            score += 15
        else:
            score += 5
            
        # Minimum balance scoring
        if min_balance >= 0:
            score += 25
        elif min_balance >= -1000:
            score += 15
        elif min_balance >= -5000:
            score += 10
        else:
            score += 0
            
        # Negative days scoring
        if negative_days == 0:
            score += 25
        elif negative_days <= 3:
            score += 15
        elif negative_days <= 7:
            score += 10
        else:
            score += 0
            
        # Volatility scoring
        if volatility <= 10:
            score += 20
        elif volatility <= 25:
            score += 15
        elif volatility <= 50:
            score += 10
        else:
            score += 0
            
        # Grade assignment
        if score >= 85:
            return 'A+'
        elif score >= 75:
            return 'A'
        elif score >= 65:
            return 'B+'
        elif score >= 55:
            return 'B'
        elif score >= 45:
            return 'C+'
        elif score >= 35:
            return 'C'
        elif score >= 25:
            return 'D+'
        else:
            return 'D'
    
    def _detect_mca_payments(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential MCA payments"""
        mca_payments = []
        
        # Look for recurring payments to potential MCA providers
        for keyword in self.mca_keywords:
            keyword_transactions = df[
                df['description'].str.contains(keyword, case=False, na=False) &
                (df['amount'] < 0)
            ].copy()
            
            if not keyword_transactions.empty:
                # Group by similar amounts (within 10% range)
                for _, transaction in keyword_transactions.iterrows():
                    amount = abs(transaction['amount'])
                    similar_amounts = keyword_transactions[
                        (abs(abs(keyword_transactions['amount']) - amount) / amount) <= 0.1
                    ]
                    
                    if len(similar_amounts) >= 3:  # At least 3 similar payments
                        payment_dates = similar_amounts['date'].tolist()
                        avg_days_between = self._calculate_avg_days_between(payment_dates)
                        
                        frequency = 'Unknown'
                        if 1 <= avg_days_between <= 3:
                            frequency = 'Daily'
                        elif 4 <= avg_days_between <= 10:
                            frequency = 'Weekly'
                        elif 11 <= avg_days_between <= 35:
                            frequency = 'Monthly'
                            
                        mca_payments.append({
                            'provider': keyword.title(),
                            'amount': amount,
                            'frequency': frequency,
                            'count': len(similar_amounts),
                            'avg_days_between': avg_days_between,
                            'first_payment': payment_dates[0],
                            'last_payment': payment_dates[-1]
                        })
                        break
        
        return mca_payments
    
    def _calculate_avg_days_between(self, dates: List[datetime]) -> float:
        """Calculate average days between payment dates"""
        if len(dates) < 2:
            return 0
            
        dates = sorted(dates)
        deltas = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
        return sum(deltas) / len(deltas) if deltas else 0

def create_visualizations(analysis: AnalysisResult):
    """Create visualizations for the analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Financial Overview")
        
        # Cash flow chart
        fig_flow = go.Figure()
        fig_flow.add_trace(go.Bar(
            x=['Deposits', 'Withdrawals'],
            y=[analysis.total_deposits, -analysis.total_withdrawals],
            marker_color=['#10b981', '#ef4444'],
            text=[f'${analysis.total_deposits:,.0f}', f'${analysis.total_withdrawals:,.0f}'],
            textposition='auto',
        ))
        fig_flow.update_layout(
            title="Cash Flow Summary",
            yaxis_title="Amount ($)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_flow, use_container_width=True)
    
    with col2:
        st.subheader("📊 Balance Trend")
        
        if not analysis.daily_balances.empty:
            fig_balance = px.line(
                analysis.daily_balances,
                x='date',
                y='balance',
                title="Daily Account Balance",
                height=400
            )
            fig_balance.update_traces(line_color='#667eea', line_width=3)
            fig_balance.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Zero Balance Line"
            )
            st.plotly_chart(fig_balance, use_container_width=True)
        else:
            st.info("Balance trend data not available")
    
    # Key metrics
    st.subheader("🎯 Key Underwriting Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average Daily Balance",
            value=f"${analysis.average_balance:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Minimum Balance",
            value=f"${analysis.minimum_balance:,.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Negative Days",
            value=f"{analysis.negative_days} days",
            delta=None
        )
    
    with col4:
        grade_color = {
            'A+': '🟢', 'A': '🟢', 'B+': '🟡', 'B': '🟡',
            'C+': '🟠', 'C': '🟠', 'D+': '🔴', 'D': '🔴', 'N/A': '⚪'
        }
        st.metric(
            label="Risk Grade",
            value=f"{grade_color.get(analysis.risk_grade, '⚪')} {analysis.risk_grade}",
            delta=None
        )

def create_mca_analysis(mca_payments: List[Dict]):
    """Create MCA payment analysis section"""
    st.subheader("🏦 MCA Payment Detection")
    
    if not mca_payments:
        st.info("✅ No recurring MCA payments detected in this statement")
        return
    
    st.warning(f"⚠️ Detected {len(mca_payments)} potential MCA provider(s)")
    
    for i, payment in enumerate(mca_payments):
        with st.expander(f"MCA Provider #{i+1}: {payment['provider']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Payment Amount", f"${payment['amount']:,.0f}")
            with col2:
                st.metric("Frequency", payment['frequency'])
            with col3:
                st.metric("Total Payments", payment['count'])
            
            st.write(f"**First Payment:** {payment['first_payment'].strftime('%B %d, %Y')}")
            st.write(f"**Last Payment:** {payment['last_payment'].strftime('%B %d, %Y')}")
            st.write(f"**Avg Days Between:** {payment['avg_days_between']:.1f} days")

def display_transaction_table(df: pd.DataFrame):
    """Display transaction table with filtering"""
    if df.empty:
        st.info("No transactions to display")
        return
    
    st.subheader("📋 Transaction Details")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_amount = st.number_input("Min Amount", value=float(df['amount'].min()))
    with col2:
        max_amount = st.number_input("Max Amount", value=float(df['amount'].max()))
    with col3:
        transaction_type = st.selectbox("Type", ["All", "Deposits", "Withdrawals"])
    
    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df['amount'] >= min_amount) & 
        (filtered_df['amount'] <= max_amount)
    ]
    
    if transaction_type == "Deposits":
        filtered_df = filtered_df[filtered_df['amount'] > 0]
    elif transaction_type == "Withdrawals":
        filtered_df = filtered_df[filtered_df['amount'] < 0]
    
    # Format for display
    display_df = filtered_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
    if 'balance' in display_df.columns:
        display_df['balance'] = display_df['balance'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export functionality
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"transactions_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Scan My Biz - Bank Statement Analyzer",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d1fae5;
        border: 1px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        color: #065f46;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        color: #92400e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 Scan My Biz</h1>
        <p>Professional Bank Statement Analysis for MCA Underwriting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        st.markdown("Upload your bank statement files for analysis")
        
        uploaded_files = st.file_uploader(
            "Choose bank statement files",
            type=['pdf', 'csv', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, CSV, TXT"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.write(f"📄 {file.name}")
        
        st.markdown("---")
        
        # System info
        st.header("🔧 System Info")
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**PDF Support:** {'✅ Available' if PDF_AVAILABLE else '❌ Missing'}")
        st.write(f"**Streamlit Version:** {st.__version__}")
        
        if not PDF_AVAILABLE:
            st.error("PDF libraries missing. Run:\n```pip install PyPDF2 pdfplumber```")
    
    # Main content area
    if not uploaded_files:
        # Welcome screen
        st.markdown("## 🎯 Welcome to Scan My Biz")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 What You'll Get:
            
            **🔍 Comprehensive Analysis**
            - Total deposits and withdrawals
            - Average daily balance calculation  
            - Days with negative balance tracking
            - Cash flow volatility assessment
            
            **⚖️ Professional Risk Assessment**
            - Automated risk grading (A+ to D)
            - Minimum balance analysis
            - Account stability scoring
            
            **🏦 MCA Detection Engine**
            - Automatic detection of existing MCA payments
            - Payment frequency analysis  
            - Provider identification
            - Risk level classification
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Advanced Features:
            
            **📋 Detailed Reporting**
            - Interactive charts and visualizations
            - Transaction-level analysis
            - Export capabilities (CSV, JSON)
            - Professional presentation format
            
            **🔧 Multi-Format Support**
            - **PDF** - Any bank's statement format
            - **CSV** - Transaction exports
            - **TXT** - Text-based statements
            
            **🚀 Enterprise Ready**
            - Batch processing multiple files
            - Robust error handling
            - Professional-grade analysis
            """)
        
        st.markdown("---")
        st.info("👈 **Get Started:** Upload your bank statement files using the sidebar to begin analysis!")
        
    else:
        # Process uploaded files
        st.header("🔄 Processing Bank Statements")
        
        parser = BankStatementParser()
        analyzer = BankStatementAnalyzer()
        
        all_transactions = []
        processing_status = st.empty()
        debug_expander = st.expander("🔍 Processing Debug Info", expanded=False)
        
        # Process each file
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            processing_status.info(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
            
            try:
                # Parse file
                transactions = parser.parse_file(file)
                
                if transactions:
                    all_transactions.extend(transactions)
                    st.success(f"✅ {file.name}: Extracted {len(transactions)} transactions")
                else:
                    st.warning(f"⚠️ {file.name}: No transactions found")
                
                # Show debug info
                with debug_expander:
                    st.write(f"**{file.name}:**")
                    for debug_msg in parser.debug_info:
                        st.code(debug_msg)
                
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
                with debug_expander:
                    st.code(f"Error in {file.name}: {str(e)}")
                    st.code(traceback.format_exc())
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        processing_status.empty()
        progress_bar.empty()
        
        if not all_transactions:
            st.error("❌ No transactions could be extracted from the uploaded files.")
            st.info("💡 **Troubleshooting Tips:**\n- Ensure files are actual bank statements\n- Check that PDFs contain text (not just images)\n- Verify CSV files have proper headers\n- Try different file formats")
            return
        
        # Perform analysis
        st.header("📊 Analysis Results")
        analysis = analyzer.analyze(all_transactions)
        
        # Display results
        if analysis.transaction_count > 0:
            # Summary stats
            st.markdown("### 📈 Executive Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Transactions", f"{analysis.transaction_count:,}")
            with col2:
                st.metric("Date Range", f"{(analysis.date_range[1] - analysis.date_range[0]).days} days")
            with col3:
                net_flow = analysis.total_deposits - analysis.total_withdrawals
                st.metric("Net Cash Flow", f"${net_flow:,.0f}")
            with col4:
                st.metric("Volatility Score", f"{analysis.volatility_score:.1f}%")
            with col5:
                risk_colors = {
                    'A+': '#10b981', 'A': '#10b981', 'B+': '#f59e0b', 'B': '#f59e0b',
                    'C+': '#ef4444', 'C': '#ef4444', 'D+': '#7c2d12', 'D': '#7c2d12'
                }
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: {risk_colors.get(analysis.risk_grade, '#6b7280')}; 
                color: white; border-radius: 8px; font-weight: bold;">
                    Risk Grade: {analysis.risk_grade}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Visualizations
            create_visualizations(analysis)
            
            st.markdown("---")
            
            # MCA Analysis
            create_mca_analysis(analysis.mca_payments)
            
            st.markdown("---")
            
            # Transaction details
            display_transaction_table(analysis.transactions)
            
            # Export functionality
            st.header("📥 Export Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export summary as JSON
                summary_data = {
                    "analysis_date": datetime.now().isoformat(),
                    "file_count": len(uploaded_files),
                    "transaction_count": analysis.transaction_count,
                    "date_range": {
                        "start": analysis.date_range[0].isoformat(),
                        "end": analysis.date_range[1].isoformat()
                    },
                    "financial_metrics": {
                        "total_deposits": analysis.total_deposits,
                        "total_withdrawals": analysis.total_withdrawals,
                        "average_balance": analysis.average_balance,
                        "minimum_balance": analysis.minimum_balance,
                        "negative_days": analysis.negative_days,
                        "volatility_score": analysis.volatility_score,
                        "risk_grade": analysis.risk_grade
                    },
                    "mca_payments": analysis.mca_payments
                }
                
                st.download_button(
                    label="📊 Download Analysis Summary (JSON)",
                    data=json.dumps(summary_data, indent=2, default=str),
                    file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Export transactions as CSV
                if not analysis.transactions.empty:
                    csv_data = analysis.transactions.to_csv(index=False)
                    st.download_button(
                        label="📋 Download All Transactions (CSV)",
                        data=csv_data,
                        file_name=f"all_transactions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                # Export daily balances as CSV
                if not analysis.daily_balances.empty:
                    balance_csv = analysis.daily_balances.to_csv(index=False)
                    st.download_button(
                        label="💰 Download Daily Balances (CSV)",
                        data=balance_csv,
                        file_name=f"daily_balances_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
        
        else:
            st.error("❌ Analysis failed - no valid transactions found")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")
        st.code(traceback.format_exc())
