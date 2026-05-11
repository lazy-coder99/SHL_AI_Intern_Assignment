import pypdf

reader = pypdf.PdfReader("SHL_AI_Intern_Assignment.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("assignment.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("PDF text extracted to assignment.txt")
