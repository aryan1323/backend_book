import numpy as np
import pandas as pd
import difflib
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

class BookOCRMatcher:
    def __init__(self, csv_path):
        """Initialize with your book dataset"""
        self.df = pd.read_csv(csv_path)
        self.titles = self.df['title'].tolist()
        
        # Load the docTR OCR model
        self.model = ocr_predictor(pretrained=True)
    
    def extract_text_from_image(self, image_path):
        """Extract text from bookshelf image using OCR"""
        doc = DocumentFile.from_images(image_path)
        result = self.model(doc)
        data = result.export()
        
        lines_info = []
        for page in data['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    line_text = ' '.join(word['value'] for word in line['words'])
                    x_coords = [float(word['geometry'][0][0]) for word in line['words']]
                    avg_x = np.mean(x_coords)
                    lines_info.append((avg_x, line_text))
        
        lines_info.sort(key=lambda x: x[0])
        
        clusters = []
        current_cluster = []
        current_x = None
        tolerance = 0.05  # adjust grouping threshold
        
        for avg_x, text in lines_info:
            if current_x is None or abs(avg_x - current_x) < tolerance:
                current_cluster.append(text)
                current_x = avg_x
            else:
                clusters.append(current_cluster)
                current_cluster = [text]
                current_x = avg_x
        
        if current_cluster:
            clusters.append(current_cluster)
        
        titles = [' '.join(cluster) for cluster in clusters]
        merged_string = ''.join(titles)
        
        return merged_string, titles
    
    def best_match(self, fragment, score_cutoff=0.4):
        matches = difflib.get_close_matches(
            fragment.lower(),
            [t.lower() for t in self.titles],
            n=1,
            cutoff=score_cutoff
        )
        if matches:
            matched_lower = matches[0]
            for i, title in enumerate(self.titles):
                if title.lower() == matched_lower:
                    score = difflib.SequenceMatcher(None, fragment.lower(), title.lower()).ratio() * 100
                    return title, i, score
        return None
    
    def segment_and_match(self, ocr_string, min_len=3, min_score=50):
        matches = []
        i = 0
        while i < len(ocr_string):
            best_match_info = None
            best_end_pos = i + min_len
            for j in range(i + min_len, min(len(ocr_string) + 1, i + 50)):
                fragment = ocr_string[i:j].strip()
                if not fragment:
                    continue
                match_result = self.best_match(fragment)
                if match_result:
                    title, idx, score = match_result
                    if best_match_info is None or score > best_match_info['score']:
                        best_match_info = {
                            'fragment': fragment,
                            'matched_title': title,
                            'score': score,
                            'categories': self.df.iloc[idx]['categories'],
                            'rating': self.df.iloc[idx]['ratings'],
                            'index': idx,
                            'start_pos': i,
                            'end_pos': j
                        }
                        best_end_pos = j
            if best_match_info and best_match_info['score'] > min_score:
                matches.append(best_match_info)
                i = best_end_pos
            else:
                i += 1
        return matches
    
    def process_bookshelf_image(self, image_path, min_score=50):
        print(f"Processing image: {image_path}")
        merged_string, raw_titles = self.extract_text_from_image(image_path)
        print(f"Extracted OCR text: {merged_string}\n")
        
        matches = self.segment_and_match(merged_string, min_score=min_score)
        
        if not matches:
            print("No book matches found.")
            return []
        
        print(f"Found {len(matches)} book matches:")
        results = []
        
        for i, match in enumerate(matches, 1):
            result = {
                'match_number': i,
                'ocr_fragment': match['fragment'],
                'matched_title': match['matched_title'],
                'categories': match['categories'],
                'rating': match['rating'],
                'confidence_score': match['score']
            }
            results.append(result)
            print(f"{i}. OCR Fragment: '{match['fragment']}'")
            print(f"   → Book Title: {match['matched_title']}")
            print(f"   → Categories: {match['categories']}")
            print(f"   → Rating: {match['rating']}")
            print(f"   → Confidence: {match['score']:.1f}%\n")
        
        return results
