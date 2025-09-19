# 📚 Bookshelf Lens

Simply snap a photo of any bookshelf, specify your preferred genres, and let **Bookshelf Lens** guide you to your next perfect read.  
No more wandering aimlessly through endless shelves!

---

## ✨ Features
- 📷 **Snap & Detect** – Upload a photo of a bookshelf, and our AI will recognize the books.  
- 🎯 **Personalized Filtering** – Choose your preferred genres, and we’ll highlight the best matches.  
- 📖 **Smart Recommendations** – Get instant suggestions tailored to your reading taste.  
- ⚡ **Fast & Simple** – Spend less time browsing, more time reading!  

---

## 🚀 How It Works
1. **Take a Photo** – Capture an image of a bookshelf.  
2. **Process with AI** – Our OCR + recognition model identifies book titles.  
3. **Filter by Genre** – Select genres you’re interested in.  
4. **Discover Reads** – Get curated book suggestions instantly.  

---

## 🛠️ Tech Stack
- **Frontend**: React / Vite  
- **Backend**: FastAPI (Python)  
- **AI/ML**: EasyOCR, YOLOv8, custom recommendation engine  
- **Database**: SQLite / PostgreSQL (configurable)  
- **Hosting**: Azure / Localhost  

---

## 📦 Installation

Clone the repo:
```bash
git clone https://github.com/your-username/bookshelf-lens.git
cd bookshelf-lens
```
Backend Setup (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload



Frontend Setup (React + Vite)
cd frontend
npm install
npm run dev
```
🌐 Usage

Open the app in your browser.

Upload a photo of a bookshelf.

Select your favorite genres.

Enjoy your personalized book recommendations!

🗺️ Roadmap

 Add support for multiple languages in OCR

 Improve book cover recognition

 Integrate with Goodreads / OpenLibrary API

 Add user profiles & reading history

🤝 Contributing

Contributions are welcome! Please fork the repo, create a new branch, and submit a PR.
