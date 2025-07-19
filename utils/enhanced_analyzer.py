import pandas as pd
import numpy as np
import re
import joblib
import time
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import textstat

warnings.filterwarnings('ignore')

class EnhancedReviewAnalyzer:
    """Enhanced ML-powered review analyzer with advanced features and better confidence scoring"""
    
    def __init__(self, use_advanced_model=True):
        self.use_advanced_model = use_advanced_model
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.linguistic_features = None
        self.model_type = "advanced" if use_advanced_model else "basic"
        self.load_models()
        
        # Initialize NLTK components
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def load_models(self):
        """Load the appropriate trained model and components"""
        try:
            if self.use_advanced_model:
                # Try to load advanced models
                if (os.path.exists('advanced_ensemble_model.pkl') and 
                    os.path.exists('advanced_tfidf_vectorizer.pkl') and
                    os.path.exists('linguistic_scaler.pkl') and
                    os.path.exists('linguistic_features.pkl')):
                    
                    self.model = joblib.load('advanced_ensemble_model.pkl')
                    self.vectorizer = joblib.load('advanced_tfidf_vectorizer.pkl')
                    self.scaler = joblib.load('linguistic_scaler.pkl')
                    self.linguistic_features = joblib.load('linguistic_features.pkl')
                    print("✅ Advanced ensemble model loaded successfully")
                    return
                else:
                    print("⚠️ Advanced models not found, falling back to basic models")
                    self.use_advanced_model = False
            
            # Load basic models
            if os.path.exists('random_forest_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
                self.model = joblib.load('random_forest_model.pkl')
                self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
                print(f"✅ Basic {self.model_type} model loaded successfully")
            else:
                raise FileNotFoundError("No model files found")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
            self.vectorizer = None
    
    def advanced_preprocessing(self, text: str) -> str:
        """Advanced text preprocessing with multiple cleaning steps"""
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
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract comprehensive linguistic features for advanced analysis"""
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(nltk.sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Readability scores
        try:
            features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            features['gunning_fog'] = textstat.gunning_fog(text)
            features['smog_index'] = textstat.smog_index(text)
            features['automated_readability_index'] = textstat.automated_readability_index(text)
        except:
            features['flesch_reading_ease'] = 0
            features['gunning_fog'] = 0
            features['smog_index'] = 0
            features['automated_readability_index'] = 0
        
        # Sentiment analysis
        try:
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = sia.polarity_scores(text)
            features['sentiment_compound'] = sentiment_scores['compound']
            features['sentiment_positive'] = sentiment_scores['pos']
            features['sentiment_negative'] = sentiment_scores['neg']
            features['sentiment_neutral'] = sentiment_scores['neu']
        except:
            features['sentiment_compound'] = 0
            features['sentiment_positive'] = 0
            features['sentiment_negative'] = 0
            features['sentiment_neutral'] = 0
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            features['textblob_polarity'] = blob.sentiment.polarity
            features['textblob_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['textblob_polarity'] = 0
            features['textblob_subjectivity'] = 0
        
        # Language complexity features
        features['unique_words_ratio'] = len(set(text.split())) / features['word_count'] if features['word_count'] > 0 else 0
        features['capitalization_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / len(text) if len(text) > 0 else 0
        
        # Fake review indicators
        fake_indicators = [
            'amazing', 'incredible', 'perfect', 'best ever', 'highly recommend',
            'excellent', 'outstanding', 'fantastic', 'wonderful', 'awesome',
            'love it', 'great product', 'fast shipping', 'good quality',
            'five stars', 'definitely recommend', 'worth every penny'
        ]
        
        genuine_indicators = [
            'however', 'but', 'although', 'despite', 'while',
            'after using', 'compared to', 'months', 'weeks', 'days',
            'pros', 'cons', 'advantages', 'disadvantages', 'issues',
            'problem', 'disappointed', 'could be better'
        ]
        
        text_lower = text.lower()
        features['fake_indicator_count'] = sum(1 for indicator in fake_indicators if indicator in text_lower)
        features['genuine_indicator_count'] = sum(1 for indicator in genuine_indicators if indicator in text_lower)
        features['fake_genuine_ratio'] = features['fake_indicator_count'] / (features['genuine_indicator_count'] + 1)
        
        return features
    
    def predict_single_review_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced prediction with detailed confidence scoring and explanations"""
        if not self.model or not self.vectorizer:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'reason': 'Model not loaded',
                'detailed_analysis': {},
                'risk_factors': [],
                'trust_indicators': []
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
                    'trust_indicators': []
                }
            
            if self.use_advanced_model and self.scaler and self.linguistic_features:
                # Use advanced model with linguistic features
                # Extract linguistic features
                linguistic_features = self.extract_linguistic_features(clean_text)
                linguistic_array = np.array([[linguistic_features[col] for col in self.linguistic_features]])
                linguistic_scaled = self.scaler.transform(linguistic_array)
                
                # Vectorize text
                text_vector = self.vectorizer.transform([clean_text])
                
                # Combine features
                combined_features = np.hstack([text_vector.toarray(), linguistic_scaled])
                
                # Get prediction and probability
                prediction = self.model.predict(combined_features)[0]
                probabilities = self.model.predict_proba(combined_features)[0]
                
            else:
                # Use basic model
                text_vector = self.vectorizer.transform([clean_text])
                prediction = self.model.predict(text_vector)[0]
                probabilities = self.model.predict_proba(text_vector)[0]
            
            # Map prediction to label
            label = 'Fake (CG)' if prediction == 'CG' else 'Genuine (OR)'
            confidence = max(probabilities) * 100
            
            # Generate detailed analysis
            detailed_analysis = self.generate_detailed_analysis(clean_text, linguistic_features if self.use_advanced_model else None)
            
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
                'trust_indicators': []
            }
    
    def generate_detailed_analysis(self, text: str, linguistic_features: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate detailed analysis of the review"""
        analysis = {}
        
        # Basic text analysis
        analysis['text_length'] = len(text)
        analysis['word_count'] = len(text.split())
        analysis['sentence_count'] = len(nltk.sent_tokenize(text))
        
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
            analysis['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            analysis['gunning_fog'] = textstat.gunning_fog(text)
            
            # Readability classification
            if analysis['flesch_reading_ease'] >= 80:
                analysis['readability_level'] = 'Very Easy'
            elif analysis['flesch_reading_ease'] >= 60:
                analysis['readability_level'] = 'Easy'
            elif analysis['flesch_reading_ease'] >= 40:
                analysis['readability_level'] = 'Moderate'
            else:
                analysis['readability_level'] = 'Difficult'
        except:
            analysis['readability_level'] = 'Unknown'
        
        # Language complexity
        analysis['unique_words_ratio'] = len(set(text.split())) / analysis['word_count'] if analysis['word_count'] > 0 else 0
        analysis['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Add linguistic features if available
        if linguistic_features:
            analysis['linguistic_features'] = linguistic_features
        
        return analysis
    
    def identify_risk_factors(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors for fake reviews"""
        risk_factors = []
        text_lower = text.lower()
        
        # Length-based risks
        if analysis['word_count'] < 5:
            risk_factors.append("Very short review (less than 5 words)")
        elif analysis['word_count'] > 200:
            risk_factors.append("Unusually long review (over 200 words)")
        
        # Sentiment-based risks
        if analysis['sentiment']['compound'] > 0.8:
            risk_factors.append("Extremely positive sentiment (potential fake enthusiasm)")
        
        # Readability risks
        if analysis.get('flesch_reading_ease', 0) < 20:
            risk_factors.append("Very difficult to read (potential AI-generated)")
        
        # Language pattern risks
        fake_patterns = [
            ('amazing', 'Generic positive language'),
            ('incredible', 'Overly enthusiastic language'),
            ('perfect', 'Absolute positive language'),
            ('best ever', 'Extreme superlative'),
            ('highly recommend', 'Common fake positive phrase'),
            ('excellent', 'Generic positive adjective'),
            ('outstanding', 'Overly positive language'),
            ('fantastic', 'Generic positive language'),
            ('wonderful', 'Generic positive language'),
            ('awesome', 'Casual overly positive language'),
            ('love it', 'Generic positive phrase'),
            ('great product', 'Generic product praise'),
            ('fast shipping', 'Generic shipping comment'),
            ('good quality', 'Generic quality statement'),
            ('five stars', 'Explicit rating mention'),
            ('definitely recommend', 'Strong recommendation language'),
            ('worth every penny', 'Generic value statement')
        ]
        
        for pattern, reason in fake_patterns:
            if pattern in text_lower:
                risk_factors.append(reason)
        
        # Repetition risks
        words = text_lower.split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.1:  # More than 10% repetition
                risk_factors.append("Excessive word repetition")
        
        return risk_factors[:5]  # Limit to top 5 risk factors
    
    def identify_trust_indicators(self, text: str, analysis: Dict[str, Any]) -> List[str]:
        """Identify trust indicators for genuine reviews"""
        trust_indicators = []
        text_lower = text.lower()
        
        # Length-based trust
        if 20 <= analysis['word_count'] <= 150:
            trust_indicators.append("Appropriate review length")
        
        # Sentiment-based trust
        if -0.3 <= analysis['sentiment']['compound'] <= 0.7:
            trust_indicators.append("Balanced sentiment (not overly positive)")
        
        # Readability trust
        if 30 <= analysis.get('flesch_reading_ease', 0) <= 80:
            trust_indicators.append("Appropriate readability level")
        
        # Language pattern trust
        genuine_patterns = [
            ('however', 'Balanced review language'),
            ('but', 'Balanced review language'),
            ('although', 'Balanced review language'),
            ('despite', 'Balanced review language'),
            ('while', 'Balanced review language'),
            ('after using', 'Specific usage experience'),
            ('compared to', 'Comparative analysis'),
            ('months', 'Long-term usage indication'),
            ('weeks', 'Extended usage period'),
            ('days', 'Specific time reference'),
            ('pros', 'Structured review'),
            ('cons', 'Structured review'),
            ('advantages', 'Structured review'),
            ('disadvantages', 'Structured review'),
            ('issues', 'Problem acknowledgment'),
            ('problem', 'Problem acknowledgment'),
            ('disappointed', 'Honest negative feedback'),
            ('could be better', 'Constructive criticism')
        ]
        
        for pattern, reason in genuine_patterns:
            if pattern in text_lower:
                trust_indicators.append(reason)
        
        # Specificity indicators
        if any(word in text_lower for word in ['bought', 'purchased', 'ordered', 'received']):
            trust_indicators.append("Purchase-specific language")
        
        if any(word in text_lower for word in ['delivery', 'shipping', 'arrived', 'packaging']):
            trust_indicators.append("Delivery experience mentioned")
        
        return trust_indicators[:5]  # Limit to top 5 trust indicators
    
    def generate_comprehensive_reason(self, text: str, prediction: str, confidence: float, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive reasoning for the prediction"""
        reasons = []
        
        # Confidence-based reasoning
        if confidence > 90:
            reasons.append("Very high model confidence")
        elif confidence > 80:
            reasons.append("High model confidence")
        elif confidence > 70:
            reasons.append("Moderate model confidence")
        else:
            reasons.append("Low model confidence - manual review recommended")
        
        # Sentiment-based reasoning
        sentiment = analysis.get('sentiment_label', 'Unknown')
        if sentiment == 'Positive' and prediction == 'CG':
            reasons.append("Overly positive sentiment detected")
        elif sentiment == 'Negative' and prediction == 'OR':
            reasons.append("Genuine negative feedback detected")
        
        # Length-based reasoning
        word_count = analysis.get('word_count', 0)
        if word_count < 10:
            reasons.append("Very short review")
        elif word_count > 100:
            reasons.append("Detailed review")
        
        # Readability-based reasoning
        readability = analysis.get('readability_level', 'Unknown')
        if readability == 'Very Easy' and prediction == 'CG':
            reasons.append("Overly simple language")
        elif readability == 'Difficult' and prediction == 'CG':
            reasons.append("Unnaturally complex language")
        
        # Risk factors
        risk_factors = self.identify_risk_factors(text, analysis)
        if risk_factors:
            reasons.append(f"Risk factors: {risk_factors[0]}")
        
        # Trust indicators
        trust_indicators = self.identify_trust_indicators(text, analysis)
        if trust_indicators:
            reasons.append(f"Trust indicators: {trust_indicators[0]}")
        
        return "; ".join(reasons[:3])  # Limit to top 3 reasons
    
    def predict_batch_reviews_enhanced(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Enhanced batch prediction for multiple reviews"""
        if not self.model or not self.vectorizer:
            return [{
                'prediction': 'Unknown', 
                'confidence': 0.0, 
                'reason': 'Model not loaded',
                'detailed_analysis': {},
                'risk_factors': [],
                'trust_indicators': []
            } for _ in texts]
        
        try:
            # Process each review individually for detailed analysis
            results = []
            for text in texts:
                result = self.predict_single_review_enhanced(text)
                results.append(result)
            
            return results
            
        except Exception as e:
            return [{
                'prediction': 'Error', 
                'confidence': 0.0, 
                'reason': f'Analysis error: {str(e)}',
                'detailed_analysis': {},
                'risk_factors': ['Technical error'],
                'trust_indicators': []
            } for _ in texts]
    
    def analyze_dataset_enhanced(self, df: pd.DataFrame, batch_size: int = 100) -> Dict[str, Any]:
        """Enhanced analysis of an entire dataset with detailed insights"""
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
        
        print(f"🔍 Analyzing {len(df)} reviews with enhanced model...")
        
        # Process in batches for memory efficiency
        all_results = []
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_texts = batch_df[text_col].tolist()
            
            print(f"   Processing batch {i//batch_size + 1}/{total_batches} ({len(batch_texts)} reviews)")
            
            batch_results = self.predict_batch_reviews_enhanced(batch_texts)
            all_results.extend(batch_results)
        
        # Generate comprehensive summary
        summary = self.generate_enhanced_summary(all_results)
        
        return {
            'results': all_results,
            'summary': summary,
            'model_type': self.model_type,
            'total_reviews': len(df)
        }
    
    def generate_enhanced_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary with advanced metrics"""
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
        
        # Risk factor analysis
        all_risk_factors = []
        for r in results:
            all_risk_factors.extend(r.get('risk_factors', []))
        
        risk_factor_counts = {}
        for risk in all_risk_factors:
            risk_factor_counts[risk] = risk_factor_counts.get(risk, 0) + 1
        
        # Trust indicator analysis
        all_trust_indicators = []
        for r in results:
            all_trust_indicators.extend(r.get('trust_indicators', []))
        
        trust_indicator_counts = {}
        for trust in all_trust_indicators:
            trust_indicator_counts[trust] = trust_indicator_counts.get(trust, 0) + 1
        
        # Sentiment analysis
        sentiments = [r.get('detailed_analysis', {}).get('sentiment_label', 'Unknown') for r in results]
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Readability analysis
        readability_levels = [r.get('detailed_analysis', {}).get('readability_level', 'Unknown') for r in results]
        readability_counts = {}
        for level in readability_levels:
            readability_counts[level] = readability_counts.get(level, 0) + 1
        
        return {
            'total_reviews': total_reviews,
            'fake_reviews': fake_count,
            'genuine_reviews': genuine_count,
            'error_count': error_count,
            'authenticity_score': (genuine_count / total_reviews * 100) if total_reviews > 0 else 0,
            'average_confidence': round(avg_confidence, 2),
            'risk_factor_analysis': dict(sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'trust_indicator_analysis': dict(sorted(trust_indicator_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'sentiment_distribution': sentiment_counts,
            'readability_distribution': readability_counts,
            'high_confidence_reviews': sum(1 for r in results if r['confidence'] > 80),
            'low_confidence_reviews': sum(1 for r in results if r['confidence'] < 60)
        }

# Import os at the top
import os 