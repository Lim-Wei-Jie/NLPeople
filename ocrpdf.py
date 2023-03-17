
# pip install ocrmypdf
# pip install pdfplumber
# pip install transformers

import ocrmypdf

# if __name__ == '__main__':  # To ensure correct behavior on Windows and macOS
#     ocrmypdf.ocr('Grab Q2 22.pdf', 'Grab Q2 22_OCR.pdf', deskew=False, force_ocr=True)



import pdf2image
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

pdf_file = ""

def pdf_to_img(pdf_file):
    return pdf2image.convert_from_path(pdf_file)


def ocr_core(file):
    text = pytesseract.image_to_string(file)
    return text


def print_pages(pdf_file):
    images = pdf_to_img(pdf_file)
    counter = 1
    for pg, img in enumerate(images):
        filename = "Grab Q2" + str(counter)
        with open(file=filename +'.txt', mode = 'w') as f:
            f.write(ocr_core(img))
        counter += 1


print_pages('Grab Q2 22.pdf')



nlp_spacy = spacy.load("en_core_web_sm")
doc = nlp_spacy("Grab Q22.txt")

sentences_annual_report = []
for sent in doc.sents:
    if len(sent.text.split()) > 6:
        print(sent.text)
        sentences_annual_report.append(sent.text)

