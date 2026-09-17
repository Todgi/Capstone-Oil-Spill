from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os
from datetime import datetime
import argparse

def create_element(name):
    return OxmlElement(name)

def create_attribute(element, name, value):
    element.set(qn(name), value)

def add_page_number(paragraph):
    run = paragraph.add_run()
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')
    run._r.append(fldChar1)

    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "PAGE"
    run._r.append(instrText)

    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'end')
    run._r.append(fldChar2)

def get_file_extension(filename):
    return os.path.splitext(filename)[1].lower()

def is_source_code_file(filename):
    """Check if file is a source code file"""
    source_extensions = {
        '.py', '.js', '.html', '.css', '.json', '.xml', '.md', 
        '.txt', '.sql', '.sh', '.bat', '.ps1', '.ts', '.jsx', 
        '.tsx', '.vue', '.php', '.java', '.cpp', '.c', '.h', 
        '.hpp', '.cs', '.go', '.rb', '.swift', '.kt', '.rs'
    }
    return get_file_extension(filename) in source_extensions

def is_image_file(filename):
    """Check if file is an image file"""
    image_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff',
        '.svg', '.webp', '.ico', '.raw', '.cr2', '.nef', '.arw'
    }
    return get_file_extension(filename) in image_extensions

def get_language_from_extension(ext):
    language_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.html': 'HTML',
        '.css': 'CSS',
        '.json': 'JSON',
        '.xml': 'XML',
        '.md': 'Markdown',
        '.txt': 'Text',
        '.sql': 'SQL',
        '.sh': 'Shell Script',
        '.bat': 'Batch Script',
        '.ps1': 'PowerShell'
    }
    return language_map.get(ext, 'Unknown')

def export_code_to_word(directory_path, output_path, exclude_dirs=None, exclude_files=None):
    """
    Export source code files to a Word document
    
    Args:
        directory_path (str): Path to the directory containing source code
        output_path (str): Path to save the Word document
        exclude_dirs (list): List of directory names to exclude
        exclude_files (list): List of file names to exclude
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'node_modules', 'venv', 'env', 'data']
    if exclude_files is None:
        exclude_files = ['.gitignore', '.env', '*.pyc']

    # Create a new Word document
    doc = Document()
    
    # Set document properties
    doc.core_properties.title = "WebGIS Source Code Documentation"
    doc.core_properties.author = "Code Exporter"
    doc.core_properties.created = datetime.now()
    
    # Add title page
    title = doc.add_heading('WebGIS Source Code Documentation', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add date
    date_paragraph = doc.add_paragraph()
    date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_run = date_paragraph.add_run(datetime.now().strftime("%d %B %Y"))
    date_run.italic = True
    
    # Add page break
    doc.add_page_break()
    
    # Add table of contents
    doc.add_heading('Table of Contents', level=1)
    toc_paragraph = doc.add_paragraph()
    toc_run = toc_paragraph.add_run()
    fldChar1 = create_element('w:fldChar')
    create_attribute(fldChar1, 'w:fldCharType', 'begin')
    toc_run._r.append(fldChar1)
    
    instrText = create_element('w:instrText')
    create_attribute(instrText, 'xml:space', 'preserve')
    instrText.text = "TOC \\o '1-3' \\h \\z \\u"
    toc_run._r.append(instrText)
    
    fldChar2 = create_element('w:fldChar')
    create_attribute(fldChar2, 'w:fldCharType', 'end')
    toc_run._r.append(fldChar2)
    
    doc.add_page_break()
    
    # Add project structure
    doc.add_heading('Project Structure', level=1)
    structure_paragraph = doc.add_paragraph()
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        level = root.replace(directory_path, '').count(os.sep)
        indent = '  ' * level
        structure_paragraph.add_run(f'{indent}{os.path.basename(root)}/\n')
        
        sub_indent = '  ' * (level + 1)
        for f in files:
            if not any(f.endswith(ext) for ext in exclude_files) and is_source_code_file(f):
                structure_paragraph.add_run(f'{sub_indent}{f}\n')
    
    doc.add_page_break()
    
    # Add source code files
    doc.add_heading('Source Code Files', level=1)
    
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # Skip excluded files and non-source code files
            if any(file.endswith(ext) for ext in exclude_files) or not is_source_code_file(file):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, directory_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add file heading
                doc.add_heading(f'File: {rel_path}', level=2)
                
                # Add language info
                ext = get_file_extension(file)
                language = get_language_from_extension(ext)
                doc.add_paragraph(f'Language: {language}')
                
                # Add code content
                code_paragraph = doc.add_paragraph()
                code_paragraph.style = 'No Spacing'
                code_run = code_paragraph.add_run(content)
                code_run.font.name = 'Courier New'
                code_run.font.size = Pt(9)
                
                # Add page break between files
                doc.add_page_break()
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
    
    # Add footer with page numbers
    section = doc.sections[0]
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(paragraph)
    
    # Save document
    doc.save(output_path)
    print(f"Document successfully generated at: {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Export source code to Word document')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing source code')
    parser.add_argument('--output', '-o', default='source_code_documentation.docx', 
                       help='Output Word document path (default: source_code_documentation.docx)')
    parser.add_argument('--exclude-dirs', nargs='+', help='Directories to exclude')
    parser.add_argument('--exclude-files', nargs='+', help='File patterns to exclude')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found")
        return
    
    # Export code to Word
    export_code_to_word(args.input, args.output, args.exclude_dirs, args.exclude_files)

if __name__ == "__main__":
    main() 