# VeriQo - AI-Powered Review Analysis Platform

VeriQo is a comprehensive AI-powered platform for analyzing product reviews, detecting fake reviews, and providing insights from expert YouTube reviews. Built with Flask, machine learning, and modern web technologies.

## 🚀 Features

### 🔍 **Fake Review Detection**
- Upload CSV files with product reviews
- AI-powered analysis using machine learning models
- Real-time detection of suspicious patterns
- Confidence scoring and detailed insights
- Interactive dashboard with charts and statistics

### 🛒 **Amazon Product Analysis**
- Analyze Amazon product reviews by URL or ASIN
- Fetch up to 200 reviews automatically
- Comprehensive analysis with AI insights
- Visual results with charts and metrics

### 📺 **YouTube Expert Review Analysis**
- Search for expert tech reviews on YouTube
- AI-powered transcript analysis
- Extract specifications, pros, cons, and recommendations
- Comprehensive reports from trusted reviewers

### 🎨 **Modern UI/UX**
- Beautiful, responsive design
- Dark/light theme support
- Interactive charts and visualizations
- Smooth animations and transitions
- Mobile-friendly interface

## 🛠️ Technology Stack

- **Backend**: Flask, Python
- **Frontend**: HTML5, TailwindCSS, Alpine.js
- **Machine Learning**: Scikit-learn, NLTK, TextBlob
- **AI Integration**: Google Gemini AI
- **Charts**: Chart.js
- **Icons**: Lucide Icons
- **Styling**: TailwindCSS with custom gradients

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/veriqo.git
cd veriqo
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
cp env_example.txt .env
```

Edit `.env` file and add your API keys:
```env
SECRET_KEY=your-secret-key-here
SERPERAPI_KEY=your-serper-api-key
GEMINI_API_KEY=your-gemini-api-key
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
VeriQo/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── env_example.txt                 # Environment variables template
├── .gitignore                      # Git ignore file
├── README.md                       # Project documentation
├── random_forest_model.pkl         # ML model
├── tfidf_vectorizer.pkl            # TF-IDF vectorizer
├── confusion_matrix.png            # Model performance
├── test_sample.csv                 # Test data
├── templates/                      # HTML templates
│   ├── base.html                   # Base template
│   ├── index.html                  # Landing page
│   ├── dashboard.html              # Analysis dashboard
│   ├── amazon_analyzer.html        # Amazon analyzer
│   ├── amazon_results.html         # Amazon results
│   ├── youtube_analyzer.html       # YouTube analyzer
│   ├── youtube_results.html        # YouTube results
│   ├── youtube_report_download.html # Report download
│   ├── login.html                  # Login page
│   └── signup.html                 # Signup page
├── static/                         # Static assets
│   ├── js/
│   │   └── charts.js               # Chart configurations
│   └── sample_reviews.csv          # Sample data
└── utils/                          # Utility modules
    ├── ml_analyzer.py              # Basic ML analyzer
    ├── enhanced_analyzer.py        # Enhanced analyzer
    ├── optimized_enhanced_analyzer.py # Optimized analyzer
    └── review_utils.py             # Utility functions
```

## 🔧 Configuration

### Environment Variables

- `SECRET_KEY`: Flask secret key for sessions
- `SERPERAPI_KEY`: Serper API key for web search
- `GEMINI_API_KEY`: Google Gemini AI API key

### API Keys Required

1. **Serper API**: For web search functionality
   - Sign up at [serper.dev](https://serper.dev)
   - Get your API key

2. **Google Gemini AI**: For AI-powered analysis
   - Sign up at [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Get your API key

## 🎯 Usage

### 1. Fake Review Detection
1. Go to the homepage
2. Upload a CSV file with review data
3. Wait for AI analysis
4. View results in the interactive dashboard

### 2. Amazon Product Analysis
1. Navigate to "Product Analysis"
2. Enter Amazon product URL or ASIN
3. Choose analysis options
4. View comprehensive results

### 3. YouTube Expert Review Analysis
1. Navigate to "Content Analysis"
2. Enter product name
3. Get AI-powered insights from expert reviews

## 📊 CSV Format

For fake review detection, your CSV should contain:
- A text column (named: `text`, `review_text`, `review`, `content`, or `Text`)
- Optional: `rating`, `author`, `date` columns

Example:
```csv
text,rating,author,date
"This product is amazing!",5,John,2024-01-15
"Not worth the money",2,Jane,2024-01-14
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Flask](https://flask.palletsprojects.com/) - Web framework
- [TailwindCSS](https://tailwindcss.com/) - CSS framework
- [Chart.js](https://www.chartjs.org/) - Charting library
- [Google Gemini AI](https://ai.google.dev/) - AI capabilities
- [Lucide Icons](https://lucide.dev/) - Icon library

## 📞 Support

If you have any questions or need help, please open an issue on GitHub or contact the development team.

---

**VeriQo** - Smart. Sharp. Verified. 🔍✨ 