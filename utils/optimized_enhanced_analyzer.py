import pandas as pd
import numpy as np
import re
import joblib
import os
import warnings
from typing import List, Dict, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat

warnings.filterwarnings('ignore')

class OptimizedEnhancedAnalyzer:
    """Memory-efficient enhanced analyzer with significant performance improvements"""
    
    def __init__(self, use_optimized_model=True):
        self.use_optimized_model = use_optimized_model
        self.model = None
        self.vectorizer = None
        self.feature_selector = None
        self.scaler = None
        self.linguistic_features = None
        self.model_type = "optimized_advanced" if use_optimized_model else "basic"
        self.load_models()
        
        # Initialize NLTK components
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def load_models(self):
        """Load the appropriate trained model and components"""
        try:
            if self.use_optimized_model:
                # Try to load optimized advanced models
                required_files = [
                    'optimized_ensemble_model.pkl',
                    'optimized_tfidf_vectorizer.pkl', 
                    'optimized_feature_selector.pkl',
                    'optimized_linguistic_scaler.pkl',
                    'optimized_linguistic_features.pkl'
                ]
                
                if all(os.path.exists(f) for f in required_files):
                    self.model = joblib.load('optimized_ensemble_model.pkl')
                    self.vectorizer = joblib.load('optimized_tfidf_vectorizer.pkl')
                    self.feature_selector = joblib.load('optimized_feature_selector.pkl')
                    self.scaler = joblib.load('optimized_linguistic_scaler.pkl')
                    self.linguistic_features = joblib.load('optimized_linguistic_features.pkl')
                    print("✅ Optimized advanced ensemble model loaded successfully")
                    return
                else:
                    print("⚠️ Optimized advanced models not found, falling back to basic models")
                    self.use_optimized_model = False
            
            # Load basic models
            if os.path.exists('random_forest_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
                self.model = joblib.load('random_forest_model.pkl')
                self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
                print(f"✅ Basic model loaded successfully")
            else:
                raise FileNotFoundError("No model files found")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
            self.vectorizer = None
    
    def advanced_preprocessing(self, text: str) -> str:
        """Advanced text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z\s\.\!\?\,\;\:\-\(\)]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_key_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract optimized key linguistic features"""
        features = {}
        
        # Basic text statistics
        words = text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Sentiment analysis
        try:
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            features['sentiment_compound'] = sentiment['compound']
            features['sentiment_positive'] = sentiment['pos']
            features['sentiment_negative'] = sentiment['neg']
        except:
            features['sentiment_compound'] = 0
            features['sentiment_positive'] = 0
            features['sentiment_negative'] = 0
        
        # Readability
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        except:
            features['flesch_reading_ease'] = 0
        
        # Language complexity
        features['unique_words_ratio'] = len(set(words)) / len(words) if words else 0
        features['capitalization_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # Fake/genuine indicators
        fake_indicators = ['amazing', 'incredible', 'perfect', 'highly recommend', 'excellent', 'fantastic', 'love it', 'great product']
        genuine_indicators = ['however', 'but', 'after using', 'compared to', 'months', 'weeks', 'problem', 'disappointed']
        
        text_lower = text.lower()
        features['fake_indicator_count'] = sum(1 for indicator in fake_indicators if indicator in text_lower)
        features['genuine_indicator_count'] = sum(1 for indicator in genuine_indicators if indicator in text_lower)
        features['fake_genuine_ratio'] = features['fake_indicator_count'] / (features['genuine_indicator_count'] + 1)
        
        return features
    
    def predict_single_review_optimized(self, text: str) -> Dict[str, Any]:
        """Optimized enhanced prediction with detailed analysis"""
        if not self.model or not self.vectorizer:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'reason': 'Model not loaded',
                'detailed_analysis': {},
                'risk_factors': [],
                'trust_indicators': [],
                'model_type': self.model_type
            }
        
        try:
            # Preprocess text
            clean_text = self.advanced_preprocessing(text)
            
            if not clean_text:
                return {
                    'prediction': 'Unknown',
                    'confidence': 0.0,
                    'reason': 'Empty text after preprocessing',
                    'detailed_analysis': {},
                    'risk_factors': ['Empty or invalid text'],
                    'trust_indicators': [],
                    'model_type': self.model_type
                }
            
            if self.use_optimized_model and self.feature_selector and self.scaler and self.linguistic_features:
                # Use optimized advanced model
                # Extract linguistic features
                linguistic_features = self.extract_key_linguistic_features(clean_text)
                linguistic_array = np.array([[linguistic_features[col] for col in self.linguistic_features]])
                linguistic_scaled = self.scaler.transform(linguistic_array)
                
                # Vectorize text
                text_vector = self.vectorizer.transform([clean_text])
                text_selected = self.feature_selector.transform(text_vector)
                
                # Combine features
                combined_features = np.hstack([text_selected.toarray(), linguistic_scaled])
                
                # Get prediction and probability
                prediction = self.model.predict(combined_features)[0]
                probabilities = self.model.predict_proba(combined_features)[0]
                
            else:
                # Use basic model
                text_vector = self.vectorizer.transform([clean_text])
                prediction = self.model.predict(text_vector)[0]
                probabilities = self.model.predict_proba(text_vector)[0]
                linguistic_features = None
            
            # Map prediction to label
            label = 'Fake (CG)' if prediction == 'CG' else 'Genuine (OR)'
            confidence = max(probabilities) * 100
            
            # Generate detailed analysis
            detailed_analysis = self.generate_detailed_analysis(clean_text, linguistic_features)
            
            # Generate risk factors and trust indicators
            risk_factors = self.identify_risk_factors(clean_text, detailed_analysis)
            trust_indicators = self.identify_trust_indicators(clean_text, detailed_analysis)
            
            # Generate comprehensive reason
            reason = self.generate_comprehensive_reason(clean_text, prediction, confidence, detailed_analysis)
            
            return {
                'prediction': label,
                'confidence': round(confidence, 2),
                'reason': reason,
                'detailed_analysis': detailed_analysis,
                'risk_factors': risk_factors,
                'trust_indicators': trust_indicators,
                'model_type': self.model_type
            }
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'reason': f'Analysis error: {str(e)}',
                'detailed_analysis': {},
                'risk_factors': ['Technical error in analysis'],
                'trust_indicators': [],
                'model_type': self.model_type
            }
    
    def generate_detailed_analysis(self, text: str, linguistic_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate detailed analysis of the review"""
        analysis = {}
        
        # Basic text analysis
        words = text.split()
        analysis['text_length'] = len(text)
        analysis['word_count'] = len(words)
        
        # Sentiment analysis
        try:
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            analysis['sentiment'] = sentiment
            
            # Sentiment classification
            if sentiment['compound'] >= 0.05:
                analysis['sentiment_label'] = 'Positive'
            elif sentiment['compound'] <= -0.05:
                analysis['sentiment_label'] = 'Negative'
            else:
                analysis['sentiment_label'] = 'Neutral'
        except:
            analysis['sentiment'] = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0}
            analysis['sentiment_label'] = 'Unknown'
        
        # Readability analysis
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            analysis['flesch_reading_ease'] = flesch_score
            
            if flesch_score >= 80:
                analysis['readability_level'] = 'Very Easy'
            elif flesch_score >= 60:
                analysis['readability_level'] = 'Easy'
            elif flesch_score >= 40:
                analysis['readability_level'] = 'Moderate'
            else:
                analysis['readability_level'] = 'Difficult'
        except:
            analysis['readability_level'] = 'Unknown'
            analysis['flesch_reading_ease'] = 0
        
        # Language complexity
        analysis['unique_words_ratio'] = len(set(words)) / len(words) if words else 0
        analysis['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Add linguistic features if available
        if linguistic_features:
            analysis['linguistic_features'] = linguistic_features
        
        return analysis
    
    def identify_risk_factors(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors for fake reviews"""
        risk_factors = []
        text_lower = text.lower()
        
        # Length-based risks
        word_count = analysis.get('word_count', 0)
        if word_count < 5:
            risk_factors.append("Very short review (less than 5 words)")
        elif word_count > 200:
            risk_factors.append("Unusually long review (over 200 words)")
        
        # Sentiment-based risks
        sentiment_compound = analysis.get('sentiment', {}).get('compound', 0)
        if sentiment_compound > 0.8:
            risk_factors.append("Extremely positive sentiment (potential fake enthusiasm)")
        
        # Readability risks
        flesch_score = analysis.get('flesch_reading_ease', 50)
        if flesch_score < 20:
            risk_factors.append("Very difficult to read (potential AI-generated)")
        
        # Pattern-based risks
        fake_patterns = [
            ('amazing', 'Generic positive language'),
            ('incredible', 'Overly enthusiastic language'),
            ('perfect', 'Absolute positive language'),
            ('highly recommend', 'Common fake positive phrase'),
            ('excellent', 'Generic positive adjective'),
            ('fantastic', 'Generic positive language'),
            ('love it', 'Generic positive phrase'),
            ('great product', 'Generic product praise')
        ]
        
        for pattern, reason in fake_patterns:
            if pattern in text_lower:
                risk_factors.append(reason)
                break  # Only add one pattern-based risk
        
        return risk_factors[:3]  # Limit to top 3 risk factors
    
    def identify_trust_indicators(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify trust indicators for genuine reviews"""
        trust_indicators = []
        text_lower = text.lower()
        
        # Length-based trust
        word_count = analysis.get('word_count', 0)
        if 20 <= word_count <= 150:
            trust_indicators.append("Appropriate review length")
        
        # Sentiment-based trust
        sentiment_compound = analysis.get('sentiment', {}).get('compound', 0)
        if -0.3 <= sentiment_compound <= 0.7:
            trust_indicators.append("Balanced sentiment (not overly positive)")
        
        # Readability trust
        flesch_score = analysis.get('flesch_reading_ease', 50)
        if 30 <= flesch_score <= 80:
            trust_indicators.append("Appropriate readability level")
        
        # Pattern-based trust
        genuine_patterns = [
            ('however', 'Balanced review language'),
            ('but', 'Balanced review language'),
            ('after using', 'Specific usage experience'),
            ('compared to', 'Comparative analysis'),
            ('months', 'Long-term usage indication'),
            ('weeks', 'Extended usage period'),
            ('problem', 'Problem acknowledgment'),
            ('disappointed', 'Honest negative feedback')
        ]
        
        for pattern, reason in genuine_patterns:
            if pattern in text_lower:
                trust_indicators.append(reason)
                break  # Only add one pattern-based trust indicator
        
        # Purchase indicators
        if any(word in text_lower for word in ['bought', 'purchased', 'ordered', 'received']):
            trust_indicators.append("Purchase-specific language")
        
        return trust_indicators[:3]  # Limit to top 3 trust indicators
    
    def generate_comprehensive_reason(self, text: str, prediction: str, confidence: float, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive reasoning for the prediction"""
        reasons = []
        
        # Confidence-based reasoning
        if confidence > 85:
            reasons.append("High model confidence")
        elif confidence > 70:
            reasons.append("Moderate model confidence")
        else:
            reasons.append("Low model confidence")
        
        # Sentiment-based reasoning
        sentiment_label = analysis.get('sentiment_label', 'Unknown')
        if sentiment_label == 'Positive' and prediction == 'CG':
            reasons.append("Overly positive sentiment detected")
        elif sentiment_label == 'Negative' and prediction == 'OR':
            reasons.append("Genuine negative feedback detected")
        
        # Model type
        if self.use_optimized_model:
            reasons.append("Enhanced ensemble model analysis")
        
        return "; ".join(reasons[:3])  # Limit to top 3 reasons
    
    def predict_batch_reviews_optimized(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Optimized batch prediction for multiple reviews"""
        if not self.model or not self.vectorizer:
            return [{
                'prediction': 'Unknown', 
                'confidence': 0.0, 
                'reason': 'Model not loaded',
                'detailed_analysis': {},
                'risk_factors': [],
                'trust_indicators': [],
                'model_type': self.model_type
            } for _ in texts]
        
        try:
            results = []
            for text in texts:
                result = self.predict_single_review_optimized(text)
                results.append(result)
            
            return results
            
        except Exception as e:
            return [{
                'prediction': 'Error', 
                'confidence': 0.0, 
                'reason': f'Analysis error: {str(e)}',
                'detailed_analysis': {},
                'risk_factors': ['Technical error'],
                'trust_indicators': [],
                'model_type': self.model_type
            } for _ in texts]
    
    def analyze_dataset_optimized(self, df: pd.DataFrame, batch_size: int = 100) -> Dict[str, Any]:
        """Optimized analysis of an entire dataset"""
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
        
        print(f"🔍 Analyzing {len(df)} reviews with optimized enhanced model...")
        
        # Process in batches
        all_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_texts = batch_df[text_col].tolist()
            
            print(f"   Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} reviews)")
            
            batch_results = self.predict_batch_reviews_optimized(batch_texts)
            all_results.extend(batch_results)
        
        # Generate summary
        summary = self.generate_optimized_summary(all_results)
        
        return {
            'results': all_results,
            'summary': summary,
            'model_type': self.model_type,
            'total_reviews': len(df)
        }
    
    def generate_optimized_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimized summary with key metrics"""
        if not results:
            return {}
        
        # Basic counts
        total_reviews = len(results)
        fake_count = sum(1 for r in results if r['prediction'] == 'Fake (CG)')
        genuine_count = sum(1 for r in results if r['prediction'] == 'Genuine (OR)')
        error_count = sum(1 for r in results if r['prediction'] in ['Error', 'Unknown'])
        
        # Confidence analysis
        confidences = [r['confidence'] for r in results if r['confidence'] > 0]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Risk and trust analysis
        all_risk_factors = []
        all_trust_indicators = []
        
        for r in results:
            all_risk_factors.extend(r.get('risk_factors', []))
            all_trust_indicators.extend(r.get('trust_indicators', []))
        
        # Count occurrences
        risk_counts = {}
        for risk in all_risk_factors:
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        trust_counts = {}
        for trust in all_trust_indicators:
            trust_counts[trust] = trust_counts.get(trust, 0) + 1
        
        # Sentiment analysis
        sentiments = [r.get('detailed_analysis', {}).get('sentiment_label', 'Unknown') for r in results]
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            'total_reviews': total_reviews,
            'fake_reviews': fake_count,
            'genuine_reviews': genuine_count,
            'error_count': error_count,
            'authenticity_score': (genuine_count / total_reviews * 100) if total_reviews > 0 else 0,
            'average_confidence': round(avg_confidence, 2),
            'top_risk_factors': dict(sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'top_trust_indicators': dict(sorted(trust_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'sentiment_distribution': sentiment_counts,
            'high_confidence_reviews': sum(1 for r in results if r['confidence'] > 80),
            'medium_confidence_reviews': sum(1 for r in results if 60 <= r['confidence'] <= 80),
            'low_confidence_reviews': sum(1 for r in results if r['confidence'] < 60),
            'model_performance_tier': 'Enhanced' if self.use_optimized_model else 'Basic'
        } 