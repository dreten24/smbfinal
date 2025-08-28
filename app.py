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
import traceback
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import sys

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
        
    def parse_file(self, file) -> List[Transaction]:
        """Parse uploaded file and extract transactions with robust error handling"""
        self.debug_info.clear()
        
        try:
            # Get file info
            file_name = getattr(file, 'name', 'unknown_file')
            file_size = getattr(file, 'size', 0)
            
            self.debug_info.append(f"Processing file: {file_name} (size: {file_size} bytes)")
            
            # Determine file type
            file_extension = Path(file_name).suffix.lower()
            
            # Reset file pointer
            if hasattr(file, 'seek'):
                file.seek(0)
            
            if file_extension == '.pdf':
                return self._parse_pdf(file)
            elif file_extension == '.csv':
                return self._parse_csv(file)
            elif file_extension == '.txt':
                return self._parse_text(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            self.debug_info.append(f"Error parsing file: {str(e)}")
            logger.error(f"File parsing error: {e}")
            raise
    
    def _parse_pdf(self, file) -> List[Transaction]:
        """Parse PDF bank statement with fallback methods"""
        if not PDF_AVAILABLE:
            raise ImportError("PDF parsing libraries not available. Please install PyPDF2 and pdfplumber.")
        
        text_content = ""
        
        # Try multiple PDF extraction methods
        try:
            # Method 1: pdfplumber
            with pdfplumber.open(file) as pdf:
                self.debug_info.append(f"PDF opened with {len(pdf.pages)} pages")
                for page_num, page in enumerate(pdf.pages[:15]):  # Limit to first 15 pages
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                            self.debug_info.append(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                    except Exception as page_error:
                        self.debug_info.append(f"Page {page_num + 1} extraction failed: {page_error}")
                        continue
                        
        except Exception as pdf_error:
            self.debug_info.append(f"pdfplumber failed: {pdf_error}")
            
            # Method 2: PyPDF2 fallback
            try:
                file.seek(0)  # Reset file pointer
                pdf_reader = PyPDF2.PdfReader(file)
                self.debug_info.append(f"PyPDF2 opened PDF with {len(pdf_reader.pages)} pages")
                
                for page_num in range(min(len(pdf_reader.pages), 15)):
                    try:
                        page_text = pdf_reader.pages[page_num].extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                    except Exception as page_error:
                        continue
                        
            except Exception as pypdf_error:
                self.debug_info.append(f"PyPDF2 also failed: {pypdf_error}")
                raise Exception(f"Both PDF parsers failed. pdfplumber: {pdf_error}, PyPDF2: {pypdf_error}")
        
        if not text_content.strip():
            raise ValueError("No readable text extracted from PDF. The PDF might be image-based or corrupted.")
        
        self.debug_info.append(f"Total extracted text: {len(text_content)} characters")
        return self._parse_text_content(text_content)
    
    def _parse_csv(self, file) -> List[Transaction]:
        """Parse CSV bank statement with robust column detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    self.debug_info.append(f"CSV loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not decode CSV file with any common encoding")
            
            self.debug_info.append(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            transactions = []
            
            # Enhanced column detection
            date_cols = ['date', 'Date', 'DATE', 'Transaction Date', 'Posted Date', 
                        'transaction_date', 'Effective Date', 'Value Date']
            desc_cols = ['description', 'Description', 'DESCRIPTION', 'Memo', 'Details', 
                        'Transaction Details', 'Payee', 'Reference']
            amount_cols = ['amount', 'Amount', 'AMOUNT', 'Transaction Amount', 'Credit', 
                          'Debit', 'Value', 'Sum']
            balance_cols = ['balance', 'Balance', 'BALANCE', 'Running Balance', 
                           'Account Balance', 'Available Balance']
            
            # Find actual columns
            date_col = self._find_column(df, date_cols)
            desc_col = self._find_column(df, desc_cols)
            amount_col = self._find_column(df, amount_cols)
            balance_col = self._find_column(df, balance_cols)
            
            if not date_col:
                raise ValueError(f"Date column not found. Available columns: {list(df.columns)}")
            if not desc_col:
                raise ValueError(f"Description column not found. Available columns: {list(df.columns)}")
            if not amount_col:
                raise ValueError(f"Amount column not found. Available columns: {list(df.columns)}")
            
            self.debug_info.append(f"Using columns - Date: {date_col}, Desc: {desc_col}, Amount: {amount_col}, Balance: {balance_col}")
            
            # Parse transactions
            successful_rows = 0
            for idx, row in df.iterrows():
                try:
                    # Parse date
                    date_val = row[date_col]
                    if pd.isna(date_val):
                        continue
                    date = pd.to_datetime(date_val)
                    
                    # Parse description
                    description = str(row[desc_col]).strip()
                    if not description or description.lower() in ['nan', 'none', '']:
                        continue
                    
                    # Parse amount
                    amount_val = str(row[amount_col])
                    if pd.isna(row[amount_col]) or amount_val.lower() in ['nan', 'none', '']:
                        continue
                    
                    # Clean amount string
                    amount_clean = re.sub(r'[^\d.-]', '', amount_val)
                    if amount_clean.count('-') > 1:
                        amount_clean = amount_clean.replace('-', '', amount_clean.count('-') - 1)
                    
                    amount = float(amount_clean) if amount_clean else 0
                    
                    # Parse balance if available
                    balance = None
                    if balance_col and not pd.isna(row[balance_col]):
                        balance_val = str(row[balance_col])
                        balance_clean = re.sub(r'[^\d.-]', '', balance_val)
                        if balance_clean:
                            balance = float(balance_clean)
                    
                    transactions.append(Transaction(
                        date=date,
                        description=description,
                        amount=amount,
                        balance=balance
                    ))
                    successful_rows += 1
                    
                except Exception as row_error:
                    self.debug_info.append(f"Skipped row {idx}: {row_error}")
                    continue
            
            self.debug_info.append(f"Successfully parsed {successful_rows} transactions from CSV")
            
            if not transactions:
                raise ValueError("No valid transactions found in CSV")
            
            return transactions
            
        except Exception as e:
            self.debug_info.append(f"CSV parsing failed: {e}")
            raise Exception(f"CSV parsing failed: {e}")
    
    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find matching column from candidates list"""
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
    
    def _parse_text(self, file) -> List[Transaction]:
        """Parse text file with encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text_content = None
            
            for encoding in encodings:
                try:
                    file.seek(0)
                    text_content = file.read().decode(encoding)
                    self.debug_info.append(f"Text file decoded with: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                raise ValueError("Could not decode text file")
            
            return self._parse_text_content(text_content)
            
        except Exception as e:
            self.debug_info.append(f"Text parsing failed: {e}")
            raise
    
    def _parse_text_content(self, text: str) -> List[Transaction]:
        """Enhanced text parsing with better pattern recognition"""
        transactions = []
        lines = text.split('\n')
        
        self.debug_info.append(f"Processing {len(lines)} lines of text")
        
        # Enhanced date patterns
        date_patterns = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})',    # MM/DD/YYYY
            r'(\d{1,2}/\d{1,2})\s',          # MM/DD (Capital One style)
            r'(\d{4}-\d{1,2}-\d{1,2})',      # YYYY-MM-DD
            r'(\d{1,2}-\d{1,2}-\d{2,4})',    # MM-DD-YYYY
            r'(\d{1,2}\.\d{1,2}\.\d{2,4})',  # MM.DD.YYYY (European)
        ]
        
        processed_transactions = 0
        
        for line_num, line in enumerate(lines):
            try:
                line = line.strip()
                if len(line) < 10:
                    continue
                
                # Skip obvious headers and footers
                skip_patterns = [
                    r'account\s+detail', r'continued\s+for\s+period', r'page\s+\d+', 
                    r'member\s+fdic', r'products\s+and\s+services', r'date.*description.*amount',
                    r'^\s*date\s+description', r'^\s*description\s+', r'total\s*$'
                ]
                
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                    continue
                
                # Find date
                date_match = None
                matched_date_str = None
                
                for pattern in date_patterns:
                    match = re.search(pattern, line)
                    if match:
                        matched_date_str = match.group(1)
                        date_match = match
                        break
                
                if not date_match:
                    continue
                
                # Parse date
                try:
                    if '/' in matched_date_str and len(matched_date_str.split('/')) == 2:
                        # MM/DD format - assume current year
                        month, day = matched_date_str.split('/')
                        date = datetime(2024, int(month), int(day))
                    else:
                        date = pd.to_datetime(matched_date_str)
                except:
                    continue
                
                # Extract remaining text after date
                remaining_text = line[date_match.end():].strip()
                
                # Find dollar amounts
                dollar_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
                dollar_matches = list(re.finditer(dollar_pattern, remaining_text))
                
                if not dollar_matches:
                    continue
                
                # Extract description (text before first dollar amount)
                first_dollar_pos = dollar_matches[0].start()
                description = remaining_text[:first_dollar_pos].strip()
                
                # Clean description
                description = re.sub(r'\s+', ' ', description)
                description = description[:100]  # Limit length
                
                if len(description) < 3:
                    # Try next line for description
                    if line_num + 1 < len(lines):
                        next_line = lines[line_num + 1].strip()
                        if next_line and len(next_line) > 3 and not re.search(r'^\d', next_line):
                            description = next_line[:100]
                
                if len(description) < 3:
                    continue
                
                # Parse amounts
                amounts = [float(match.group(1).replace(',', '')) for match in dollar_matches]
                
                if len(amounts) == 1:
                    transaction_amount = amounts[0]
                    balance = None
                elif len(amounts) >= 2:
                    # Usually: transaction amount, then ending balance
                    transaction_amount = amounts[-2]  # Second to last
                    balance = amounts[-1]  # Last amount
                else:
                    continue
                
                # Determine transaction sign
                transaction_amount = self._determine_transaction_sign(
                    transaction_amount, description, line
                )
                
                transactions.append(Transaction(
                    date=date,
                    description=description,
                    amount=transaction_amount,
                    balance=balance
                ))
                
                processed_transactions += 1
                
            except Exception as line_error:
                self.debug_info.append(f"Error processing line {line_num}: {line_error}")
                continue
        
        self.debug_info.append(f"Processed {processed_transactions} transactions from text")
        
        if not transactions:
            raise ValueError("No transactions could be parsed from the text content")
        
        return sorted(transactions, key=lambda t: t.date)
    
    def _determine_transaction_sign(self, amount: float, description: str, full_line: str) -> float:
        """Determine if transaction should be positive (deposit) or negative (withdrawal)"""
        desc_lower = description.lower()
        line_lower = full_line.lower()
        
        # Positive indicators (deposits/credits)
        positive_keywords = [
            'deposit', 'credit', 'uber', 'doordash', 'square', 'paypal', 'venmo', 'zelle',
            'transfer in', 'refund', 'interest', 'dividend', 'salary', 'wage', 'bonus',
            'ach deposit', 'direct deposit', 'merch bnkcd', 'fdms', 'customer deposit',
            'cash deposit', 'mobile deposit', 'wire in'
        ]
        
        # Negative indicators (withdrawals/debits)
        negative_keywords = [
            'withdrawal', 'payment', 'purchase', 'check', 'fee', 'charge', 'debit',
            'atm', 'transfer out', 'loan', 'ach withdrawal', 'wire out', 'auto pay',
            'bill pay', 'overdraft', 'maintenance', 'service charge'
        ]
        
        # Check for positive indicators
        if any(keyword in desc_lower or keyword in line_lower for keyword in positive_keywords):
            return abs(amount)
        
        # Check for negative indicators
        elif any(keyword in desc_lower or keyword in line_lower for keyword in negative_keywords):
            return -abs(amount)
        
        # Default assumption for unclear transactions (most are withdrawals)
        else:
            return -abs(amount)

class BankStatementAnalyzer:
    """Enhanced analyzer with better error handling"""
    
    def __init__(self):
        self.mca_keywords = [
            'ondeck', 'kabbage', 'bluevine', 'funding circle', 'square capital',
            'paypal working capital', 'merchant cash', 'advance', 'capify',
            'credibly', 'lendio', 'forward financing', 'rapid finance'
        ]
    
    def analyze(self, transactions: List[Transaction]) -> AnalysisResult:
        """Perform comprehensive analysis with enhanced error handling"""
        if not transactions:
            raise ValueError("No transactions provided for analysis")
        
        try:
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
            
            # Validate data
            df = df.dropna(subset=['date', 'description', 'amount'])
            
            if df.empty:
                raise ValueError("No valid transactions after data cleaning")
            
            # Calculate metrics
            deposits = df[df['amount'] > 0]['amount'].sum()
            withdrawals = abs(df[df['amount'] < 0]['amount'].sum())
            
            # Daily balances
            daily_balances = self._calculate_daily_balances(df)
            
            if daily_balances.empty:
                raise ValueError("Could not calculate daily balances")
            
            # Negative days calculation
            negative_days = len(daily_balances[daily_balances['balance'] < 0])
            
            # Other metrics
            avg_balance = daily_balances['balance'].mean()
            min_balance = daily_balances['balance'].min()
            volatility = self._calculate_volatility(daily_balances)
            risk_grade = self._calculate_risk_grade(avg_balance, negative_days, len(daily_balances))
            mca_payments = self._detect_mca_payments(df)
            
            return AnalysisResult(
                total_deposits=float(deposits),
                total_withdrawals=float(withdrawals),
                average_balance=float(avg_balance),
                minimum_balance=float(min_balance),
                negative_days=int(negative_days),
                transaction_count=len(df),
                date_range=(df['date'].min(), df['date'].max()),
                volatility_score=float(volatility),
                risk_grade=risk_grade,
                mca_payments=mca_payments,
                daily_balances=daily_balances,
                transactions=df
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise Exception(f"Analysis failed: {e}")
    
    def _calculate_daily_balances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily ending balances with error handling"""
        try:
            # Sort by date first
            df_sorted = df.sort_values('date')
            
            # If balance column exists and has data, use it
            if 'balance' in df.columns and df['balance'].notna().any():
                # Group by date and take the last balance for each day
                daily_balances = df_sorted.groupby(df_sorted['date'].dt.date).agg({
                    'balance': 'last'
                }).reset_index()
                daily_balances.columns = ['date', 'balance']
                
                # Remove rows with NaN balances
                daily_balances = daily_balances.dropna(subset=['balance'])
            else:
                # Calculate running balance from transactions
                # Estimate starting balance
                first_day_transactions = df_sorted[df_sorted['date'] == df_sorted['date'].min()]
                estimated_start = max(10000, abs(first_day_transactions['amount'].sum()) * 10)
                
                df_sorted = df_sorted.copy()
                df_sorted['running_balance'] = df_sorted['amount'].cumsum() + estimated_start
                
                daily_balances = df_sorted.groupby(df_sorted['date'].dt.date).agg({
                    'running_balance': 'last'
                }).reset_index()
                daily_balances.columns = ['date', 'balance']
            
            return daily_balances.sort_values('date')
            
        except Exception as e:
            logger.error(f"Daily balance calculation error: {e}")
            # Return minimal valid data
            return pd.DataFrame({
                'date': [datetime.now().date()],
                'balance': [0.0]
            })
    
    def _calculate_volatility(self, daily_balances: pd.DataFrame) -> float:
        """Calculate volatility score with error handling"""
        try:
            if len(daily_balances) < 2:
                return 0.0
            
            balances = daily_balances['balance']
            mean_balance = balances.mean()
            std_balance = balances.std()
            
            if pd.isna(mean_balance) or pd.isna(std_balance) or mean_balance == 0:
                return 0.0
            
            volatility = min(100.0, (std_balance / abs(mean_balance)) * 100)
            return round(float(volatility), 1)
            
        except Exception:
            return 0.0
    
    def _calculate_risk_grade(self, avg_balance: float, negative_days: int, total_days: int) -> str:
        """Calculate risk grade with bounds checking"""
        try:
            if pd.isna(avg_balance) or total_days == 0:
                return 'C'
            
            score = 85
            
            if avg_balance < 5000:
                score -= 25
            elif avg_balance < 15000:
                score -= 10
            elif avg_balance > 50000:
                score += 10
            
            negative_ratio = negative_days / total_days
            if negative_ratio > 0.2:
                score -= 30
            elif negative_ratio > 0.1:
                score -= 15
            
            # Convert to grade
            grade_map = [
                (95, 'A+'), (90, 'A'), (85, 'A-'), (80, 'B+'),
                (75, 'B'), (70, 'B-'), (65, 'C+'), (60, 'C')
            ]
            
            for threshold, grade in grade_map:
                if score >= threshold:
                    return grade
            
            return 'D'
            
        except Exception:
            return 'C'
    
    def _detect_mca_payments(self, df: pd.DataFrame) -> List[Dict]:
        """Detect MCA payments with error handling"""
        try:
            mca_payments = []
            outgoing_payments = df[df['amount'] < -100]
            
            if outgoing_payments.empty:
                return []
            
            for keyword in self.mca_keywords:
                try:
                    matching_payments = outgoing_payments[
                        outgoing_payments['description'].str.contains(keyword, case=False, na=False)
                    ]
                    
                    if len(matching_payments) >= 2:
                        avg_amount = matching_payments['amount'].mean()
                        frequency = self._calculate_payment_frequency(matching_payments)
                        
                        mca_payments.append({
                            'provider': keyword.title(),
                            'average_amount': float(avg_amount),
                            'frequency': frequency,
                            'count': len(matching_payments),
                            'risk_level': 'High' if abs(avg_amount) > 1000 else 'Medium'
                        })
                except Exception:
                    continue
            
            return sorted(mca_payments, key=lambda x: abs(x['average_amount']), reverse=True)
            
        except Exception:
            return []
    
    def _calculate_payment_frequency(self, payments: pd.DataFrame) -> str:
        """Calculate payment frequency with error handling"""
        try:
            if len(payments) < 2:
                return 'Unknown'
            
            dates = sorted(payments['date'].tolist())
            intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            
            if not intervals:
                return 'Unknown'
            
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
                
        except Exception:
            return 'Unknown'

def main():
    """Main Streamlit application with enhanced error handling"""
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
        font-family: monospace;
        font-size: 0.8rem;
    }
    .stFileUploader > div > div > div > div {
        text-align: center;
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
    
    # Display system info for debugging
    with st.expander("System Information", expanded=False):
        st.text(f"Python version: {sys.version}")
        st.text(f"PDF libraries available: {PDF_AVAILABLE}")
        st.text(f"Streamlit version: {st.__version__}")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload Bank Statements")
        st.markdown("Supported formats: PDF, CSV, TXT")
        
        # File uploader with better error handling
        try:
            uploaded_files = st.file_uploader(
                "Choose bank statement files",
                type=['pdf', 'csv', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF, CSV, or TXT bank statements from any bank",
                key="file_uploader"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
                
                # Show file details
                for file in uploaded_files:
                    st.text(f"📄 {file.name} ({file.size:,} bytes)")
                
                # Analysis button
                if st.button("🚀 Analyze Statements", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Processing bank statements..."):
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Initialize components
                            parser = BankStatementParser()
                            analyzer = BankStatementAnalyzer()
                            
                            all_transactions = []
                            debug_info = []
                            
                            # Process each file
                            for i, file in enumerate(uploaded_files):
                                try:
                                    status_text.text(f"Processing {file.name}...")
                                    progress_bar.progress((i + 1) / len(uploaded_files) * 0.7)
                                    
                                    transactions = parser.parse_file(file)
                                    all_transactions.extend(transactions)
                                    debug_info.extend(parser.debug_info)
                                    
                                except Exception as file_error:
                                    st.error(f"❌ Error processing {file.name}: {file_error}")
                                    debug_info.append(f"File {file.name} failed: {file_error}")
                                    continue
                            
                            if not all_transactions:
                                st.error("❌ No transactions found in any uploaded files")
                                st.session_state.debug_info = debug_info
                                return
                            
                            # Analyze transactions
                            status_text.text("Analyzing transactions...")
                            progress_bar.progress(0.9)
                            
                            result = analyzer.analyze(all_transactions)
                            st.session_state.analysis_result = result
                            st.session_state.debug_info = debug_info
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Analysis completed!")
                            
                            st.success(f"🎉 Successfully analyzed {len(all_transactions)} transactions!")
                            
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.session_state.debug_info.append(f"Analysis error: {e}")
                        logger.error(f"Analysis error: {e}")
                        st.code(traceback.format_exc())
            else:
                st.info("👆 Please upload bank statement files to begin analysis")
                
        except Exception as upload_error:
            st.error(f"❌ File upload error: {upload_error}")
            logger.error(f"Upload error: {upload_error}")
    
    # Main content area
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Debug information
        with st.expander("🔍 Processing Log", expanded=False):
            st.markdown('<div class="debug-panel">', unsafe_allow_html=True)
            for info in st.session_state.debug_info:
                st.text(f"• {info}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Key Metrics Dashboard
        st.header("📊 Financial Analysis Dashboard")
        
        # Primary metrics row
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
        
        # Secondary metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Total Withdrawals", f"${result.total_withdrawals:,.0f}")
        
        with col6:
            st.metric("Minimum Balance", f"${result.minimum_balance:,.0f}")
        
        with col7:
            st.metric("Volatility Score", f"{result.volatility_score}%")
        
        with col8:
            st.metric("Total Transactions", f"{result.transaction_count:,}")
        
        # Balance trend chart
        st.header("📈 Account Balance Analysis")
        
        try:
            if not result.daily_balances.empty:
                fig = px.line(
                    result.daily_balances,
                    x='date',
                    y='balance',
                    title='Daily Account Balance Trend',
                    labels={'balance': 'Account Balance ($)', 'date': 'Date'}
                )
                fig.update_traces(line=dict(color='#667eea', width=3))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Date",
                    yaxis_title="Balance ($)",
                    yaxis_tickformat="$,.0f"
                )
                # Add horizontal line at $0 for negative balance reference
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No balance data available for chart")
        except Exception as chart_error:
            st.error(f"❌ Chart error: {chart_error}")
        
        # MCA Analysis
        if result.mca_payments:
            st.header("🔍 MCA Payment Detection")
            mca_df = pd.DataFrame(result.mca_payments)
            st.dataframe(mca_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No MCA payments detected in the analysis period")
        
        # Transaction details
        st.header("💳 Transaction Summary")
        
        try:
            if not result.transactions.empty:
                # Show summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    deposit_count = len(result.transactions[result.transactions['amount'] > 0])
                    st.metric("Deposit Transactions", deposit_count)
                with col2:
                    withdrawal_count = len(result.transactions[result.transactions['amount'] < 0])
                    st.metric("Withdrawal Transactions", withdrawal_count)
                
                # Recent transactions table
                recent_transactions = result.transactions.sort_values('date', ascending=False).head(50)
                
                # Format the dataframe for display
                display_df = recent_transactions[['date', 'description', 'amount', 'balance']].copy()
                display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
                display_df['balance'] = display_df['balance'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ No transaction data available")
        except Exception as table_error:
            st.error(f"❌ Table error: {table_error}")
        
        # Export functionality
        st.header("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                csv_data = result.transactions.to_csv(index=False)
                st.download_button(
                    label="📊 Download Transactions CSV",
                    data=csv_data,
                    file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as csv_error:
                st.error(f"CSV export error: {csv_error}")
        
        with col2:
            try:
                summary = {
                    "analysis_date": datetime.now().isoformat(),
                    "file_count": len(uploaded_files) if 'uploaded_files' in locals() else 0,
                    "total_deposits": float(result.total_deposits),
                    "total_withdrawals": float(result.total_withdrawals),
                    "average_balance": float(result.average_balance),
                    "minimum_balance": float(result.minimum_balance),
                    "negative_days": int(result.negative_days),
                    "risk_grade": result.risk_grade,
                    "volatility_score": float(result.volatility_score),
                    "transaction_count": int(result.transaction_count),
                    "mca_payments": result.mca_payments
                }
                
                summary_json = json.dumps(summary, indent=2, default=str)
                st.download_button(
                    label="📋 Download Analysis Report",
                    data=summary_json,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as json_error:
                st.error(f"JSON export error: {json_error}")
        
        with col3:
            st.info("💡 Additional export formats available in premium version")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 👋 Welcome to Scan My Biz!
        
        **Professional Bank Statement Analysis for MCA Underwriting**
        
        ### What You'll Get:
        
        ✅ **Accurate Financial Metrics**
        - Total deposits and withdrawals
        - Average daily balance calculation
        - Days with negative balance (< $0.00)
        - Cash flow volatility analysis
        
        ✅ **Risk Assessment**
        - Professional risk grading (A+ to D)
        - Minimum balance tracking
        - Account performance scoring
        
        ✅ **MCA Detection**
        - Automatic detection of merchant cash advance payments
        - Payment frequency analysis
        - Risk level classification
        
        ✅ **Professional Reporting**
        - Interactive charts and visualizations
        - Detailed transaction analysis
        - Export capabilities (CSV, JSON)
        
        ### Supported File Formats:
        - **PDF** - Bank statements from any institution
        - **CSV** - Transaction exports
        - **TXT** - Text-based statements
        
        ### Get Started:
        Upload your bank statement files using the sidebar to begin your analysis!
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"Application error: {e}")
        st.code(traceback.format_exc())