import pandas as pd
import numpy as np
import re
import joblib
from datetime import datetime
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class ReviewAnalyzer:
    """Optimized ML-powered review analyzer for detecting fake reviews"""
    
    def __init__(self, use_ultrafast=False):
        self.model = None
        self.vectorizer = None
        self.use_ultrafast = use_ultrafast
        self.model_type = "ultrafast" if use_ultrafast else "optimized"
        self.load_models()
    
    def load_models(self):
        """Load the trained model and TF-IDF vectorizer"""
        try:
            if self.use_ultrafast:
                # Try to load ultra-fast models first
                if os.path.exists('ultrafast_model.pkl') and os.path.exists('ultrafast_vectorizer.pkl'):
                    self.model = joblib.load('ultrafast_model.pkl')
                    self.vectorizer = joblib.load('ultrafast_vectorizer.pkl')
                    print("✅ Ultra-fast ML models loaded successfully")
                    return
                else:
                    print("⚠️ Ultra-fast models not found, falling back to optimized models")
            
            # Load standard optimized models
            if os.path.exists('random_forest_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
                self.model = joblib.load('random_forest_model.pkl')
                self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
                print(f"✅ {self.model_type} ML models loaded successfully")
            else:
                raise FileNotFoundError("Model files not found")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
            self.vectorizer = None
    
    def preprocess_text(self, text):
        """
        Preprocess text data:
        - Convert to lowercase
        - Remove punctuation, numbers, and extra spaces
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Optimized batch preprocessing of texts"""
        return [self.preprocess_text(text) for text in texts]
    
    def predict_single_review(self, text):
        """Predict if a single review is fake or genuine"""
        if not self.model or not self.vectorizer:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'reason': 'Model not loaded'
            }
        
        try:
            # Preprocess text
            clean_text = self.preprocess_text(text)
            
            if not clean_text:
                return {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'reason': 'Empty text after preprocessing'
                }
            
            # Vectorize text
            text_vector = self.vectorizer.transform([clean_text])
            
            # Get prediction and probability
            prediction = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]
            
            # Map prediction to label
            label = 'Fake (CG)' if prediction == 'CG' else 'Genuine (OR)'
            confidence = max(probabilities) * 100
            
            # Generate reason based on features
            reason = self.generate_reason(clean_text, prediction, confidence)
            
            return {
                'prediction': label,
                'confidence': round(confidence, 2),
                'reason': reason
            }
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'reason': f'Analysis error: {str(e)}'
            }
    
    def predict_batch_reviews(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Optimized batch prediction for multiple reviews"""
        if not self.model or not self.vectorizer:
            return [{'prediction': 'Unknown', 'confidence': 0.0, 'reason': 'Model not loaded'} for _ in texts]
        
        try:
            # Batch preprocessing
            clean_texts = self.preprocess_batch(texts)
            
            # Filter out empty texts and keep track of indices
            valid_texts = []
            valid_indices = []
            for i, text in enumerate(clean_texts):
                if text:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [{'prediction': 'Unknown', 'confidence': 0.0, 'reason': 'Empty text after preprocessing'} for _ in texts]
            
            # Batch vectorization - MUCH faster than individual transforms
            text_vectors = self.vectorizer.transform(valid_texts)
            
            # Batch prediction - MUCH faster than individual predictions
            predictions = self.model.predict(text_vectors)
            probabilities = self.model.predict_proba(text_vectors)
            
            # Prepare results for all texts
            results = []
            valid_idx = 0
            
            for i, (original_text, clean_text) in enumerate(zip(texts, clean_texts)):
                if i in valid_indices:
                    prediction = predictions[valid_idx]
                    proba = probabilities[valid_idx]
                    
                    # Map prediction to label
                    label = 'Fake (CG)' if prediction == 'CG' else 'Genuine (OR)'
                    confidence = max(proba) * 100
                    
                    # Generate reason
                    reason = self.generate_reason(clean_text, prediction, confidence)
                    
                    results.append({
                        'prediction': label,
                        'confidence': round(confidence, 2),
                        'reason': reason
                    })
                    valid_idx += 1
                else:
                    results.append({
                        'prediction': 'Unknown',
                        'confidence': 0.0,
                        'reason': 'Empty text after preprocessing'
                    })
            
            return results
            
        except Exception as e:
            return [{'prediction': 'Error', 'confidence': 0.0, 'reason': f'Analysis error: {str(e)}'} for _ in texts]
    
    def generate_reason(self, text, prediction, confidence):
        """Generate a human-readable reason for the prediction"""
        reasons = []
        
        # Length-based reasons
        if len(text.split()) < 5:
            reasons.append("Very short review")
        elif len(text.split()) > 100:
            reasons.append("Unusually long review")
        
        # Common fake review patterns
        fake_patterns = [
            ('great product', 'Generic positive language'),
            ('highly recommend', 'Common fake positive phrase'),
            ('fast shipping', 'Generic shipping comment'),
            ('excellent value', 'Generic value statement'),
            ('five stars', 'Explicit rating mention'),
            ('best product ever', 'Extreme positive language')
        ]
        
        genuine_patterns = [
            ('after using', 'Specific usage experience'),
            ('compared to', 'Comparative analysis'),
            ('however', 'Balanced review'),
            ('but', 'Balanced review'),
            ('months', 'Long-term usage'),
            ('weeks', 'Extended usage period')
        ]
        
        # Check for patterns
        for pattern, reason in fake_patterns:
            if pattern in text.lower():
                reasons.append(reason)
                break
        
        for pattern, reason in genuine_patterns:
            if pattern in text.lower():
                reasons.append(reason)
                break
        
        # Confidence-based reasons
        if confidence > 85:
            reasons.append("High model confidence")
        elif confidence < 60:
            reasons.append("Low model confidence")
        
        # Default reason if none found
        if not reasons:
            if prediction == 'CG':
                reasons.append("Language patterns suggest automated generation")
            else:
                reasons.append("Language patterns suggest genuine human writing")
        
        return "; ".join(reasons[:3])  # Limit to top 3 reasons
    
    def analyze_dataset(self, df, batch_size: int = 1000):
        """Optimized analysis of an entire dataset using batch processing"""
        if not self.model or not self.vectorizer:
            raise Exception("ML models not loaded")
        
        # Find text column (flexible column names)
        text_columns = ['text', 'text_', 'review_text', 'review', 'content', 'Text']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            raise Exception(f"No text column found. Available columns: {list(df.columns)}")
        
        # Find other useful columns
        user_col = None
        timestamp_col = None
        
        for col in df.columns:
            if 'user' in col.lower() or 'id' in col.lower():
                
                user_col = col
            elif 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
        
        # Extract texts for batch processing
        texts = df[text_col].tolist()
        
        print(f"Processing {len(texts)} reviews in batches of {batch_size}...")
        
        # Process in batches for memory efficiency
        all_predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_predictions = self.predict_batch_reviews(batch_texts)
            all_predictions.extend(batch_predictions)
            
            # Progress indicator
            if (i + batch_size) % (batch_size * 5) == 0 or i + batch_size >= len(texts):
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} reviews...")
        
        # Combine results with original data
        results = []
        for idx, (row, prediction_result) in enumerate(zip(df.itertuples(), all_predictions)):
            text = getattr(row, text_col)
            
            result = {
                'index': idx,
                'text': str(text)[:200] + '...' if len(str(text)) > 200 else str(text),
                'full_text': str(text),
                'user': getattr(row, user_col, f'User_{idx}') if user_col else f'User_{idx}',
                'timestamp': getattr(row, timestamp_col, 'N/A') if timestamp_col else 'N/A',
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'reason': prediction_result['reason']
            }
            results.append(result)
        
        print(f"✅ Analysis complete! Processed {len(results)} reviews.")
        return results
    
    def analyze_dataset_parallel(self, df, n_workers: int = 4, batch_size: int = 1000):
        """Parallel processing version for very large datasets"""
        if not self.model or not self.vectorizer:
            raise Exception("ML models not loaded")
        
        # Find text column
        text_columns = ['text', 'text_', 'review_text', 'review', 'content', 'Text']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            raise Exception(f"No text column found. Available columns: {list(df.columns)}")
        
        # Find other useful columns
        user_col = None
        timestamp_col = None
        
        for col in df.columns:
            if 'user' in col.lower() or 'id' in col.lower():
                user_col = col
            elif 'time' in col.lower() or 'date' in col.lower():
                timestamp_col = col
        
        texts = df[text_col].tolist()
        print(f"Processing {len(texts)} reviews with {n_workers} parallel workers...")
        
        # Split data into chunks for parallel processing
        chunk_size = max(batch_size, len(texts) // n_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Process chunks in parallel
        all_predictions = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_chunk = {executor.submit(self.predict_batch_reviews, chunk): chunk for chunk in chunks}
            
            for i, future in enumerate(future_to_chunk):
                chunk_predictions = future.result()
                all_predictions.extend(chunk_predictions)
                print(f"Completed chunk {i + 1}/{len(chunks)}")
        
        # Combine results with original data
        results = []
        for idx, (row, prediction_result) in enumerate(zip(df.itertuples(), all_predictions)):
            text = getattr(row, text_col)
            
            result = {
                'index': idx,
                'text': str(text)[:200] + '...' if len(str(text)) > 200 else str(text),
                'full_text': str(text),
                'user': getattr(row, user_col, f'User_{idx}') if user_col else f'User_{idx}',
                'timestamp': getattr(row, timestamp_col, 'N/A') if timestamp_col else 'N/A',
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'reason': prediction_result['reason']
            }
            results.append(result)
        
        print(f"✅ Parallel analysis complete! Processed {len(results)} reviews.")
        return results
    
    def generate_summary_stats(self, results):
        """Generate summary statistics from analysis results"""
        total_reviews = len(results)
        fake_count = sum(1 for r in results if 'Fake' in r['prediction'])
        genuine_count = sum(1 for r in results if 'Genuine' in r['prediction'])
        unknown_count = total_reviews - fake_count - genuine_count
        
        avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] > 0])
        
        # Time-based analysis (if timestamps available)
        timeline_data = {}
        for result in results:
            timestamp = result['timestamp']
            if timestamp != 'N/A':
                try:
                    # Try to parse various date formats
                    if isinstance(timestamp, str):
                        date = pd.to_datetime(timestamp).date()
                    else:
                        date = timestamp.date() if hasattr(timestamp, 'date') else 'Unknown'
                    
                    date_str = str(date)
                    if date_str not in timeline_data:
                        timeline_data[date_str] = {'fake': 0, 'genuine': 0}
                    
                    if 'Fake' in result['prediction']:
                        timeline_data[date_str]['fake'] += 1
                    elif 'Genuine' in result['prediction']:
                        timeline_data[date_str]['genuine'] += 1
                except:
                    pass
        
        return {
            'total_reviews': total_reviews,
            'fake_count': fake_count,
            'genuine_count': genuine_count,
            'unknown_count': unknown_count,
            'fake_percentage': round((fake_count / total_reviews) * 100, 2) if total_reviews > 0 else 0,
            'genuine_percentage': round((genuine_count / total_reviews) * 100, 2) if total_reviews > 0 else 0,
            'average_confidence': round(avg_confidence, 2) if avg_confidence else 0,
            'timeline_data': timeline_data
        }
    
    def export_results(self, results, filename='analysis_results.csv'):
        """Export analysis results to CSV"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        return filename 