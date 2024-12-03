# 원문 : "test_loader_universal.ipynb", "test_loader.ipynb"
# 불러 사용하는 방법 
# from my_common.my_llm import *

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import RecursiveUrlLoader

from io import BytesIO
from zipfile import ZipFile
import requests
import json
from pathlib import Path
import os
import re


ALLOW_TEXT = [".txt", ".md"]
ALLOW_HTML = [".html", ".htm"]
ALLOW_PDF = [".pdf"]
ALLOW_CSV = [".csv"]
ALLOW_JSON = [".json"]
ALLOW_ZIP = [".zip"]
ALLOW_DOCX = [".docx"]

##################################################
def universal_file_loader(source):
    filename, file_extension = os.path.splitext(source)
    file_extension = file_extension.lower()
    #print(f"filename= {filename}, extension= {file_extension}")
    
    data, loader = '', None
    if file_extension in ALLOW_TEXT :
        loader = TextLoader(source)
    elif file_extension in ALLOW_HTML:
        loader = BSHTMLLoader(source)
    elif file_extension in ALLOW_PDF:
        loader = PyPDFLoader(source)
    elif file_extension in ALLOW_CSV:
        loader = CSVLoader(file_path=source)
    elif file_extension in ALLOW_DOCX:
        loader = Docx2txtLoader(file_path=source)
    elif file_extension in ALLOW_JSON:
        #loader = JSONLoader(file_path=source,
        #                    jq_schema='.content',
        #                    #text_content=False,
        #                    json_lines=True)
        data = json.loads(Path(source).read_text())
    else:
        print("Not recognize file type :", file_extension)
    if loader:
        data = loader.load()
    return data



# url loader
# "http", "https", "ftp"
##################################################
def universal_url_loader(source):
    loader = RecursiveUrlLoader(
        source,
        prevent_outside=True,
        #base_url="https://docs.python.org",
        link_regex=r'<a\s+(?:[^>]*?\s+)?href="([^"]*(?=index)[^"]*)"',
        #exclude_dirs=['https://docs.python.org/3.9/faq']
    )
    docs = loader.load()
    return docs


##################################################
def universal_zip_loader(file_url):
    url = requests.get(file_url)
    zipfile = ZipFile(BytesIO(url.content))
    zipfile.extractall("/tmp")
    docs = []
    for file_name in zipfile.namelist():
        doc = universal_file_loader("/tmp/"+file_name)
        if doc:
            docs.extend(doc)
    return docs

##################################################
def universal_dir_loader(source):
    #print("*start=",source)
    #c_dir = os.getcwd() 
    dir_list = os.listdir(source) # 현재 폴더에 있는것만 읽어 들임.
    docs = []
    dir_list.sort()
    for file_name in dir_list:
        doc = None
        if file_name[0] == ".":
            continue
        #print("name=",file_name)
        if os.path.isdir(file_name):
            doc = universal_dir_loader(source+"/"+file_name)
        else:
            doc = universal_file_loader(source+"/"+file_name)
        if doc:
            docs.extend(doc)
    return docs

##################################################

regex = ("((http|https)://)(www.)?" + "[a-zA-Z0-9@:%._\\+~#?&//=]" +
             "{2,256}\\.[a-z]" + "{2,6}\\b([-a-zA-Z0-9@:%" + "._\\+~#?&//=]*)")
p = re.compile(regex)

# 자동으로 체크해서, type을 결정하도록 수정
def universal_loader_v2(source):
    docs = []
    if(re.search(p, source)):
        docs = universal_url_loader(source)
    elif os.path.isdir(source):
        docs = universal_dir_loader(source)
    elif os.path.isfile(source):
        docs = universal_file_loader(source)        
    else:
        filename, file_extension = os.path.splitext(source)
        file_extension = file_extension.lower()
        if file_extension in ALLOW_ZIP:
            docs = universal_zip_loader(source)
    return docs

# type을 수동으로 지정해주면 사용
def universal_loader_v1(source, type='file'):
    docs = []
    if type == 'file':
        docs = universal_file_loader(source)
    elif type == 'dir':
        docs = universal_dir_loader(source)
    elif type == 'url':
        docs = universal_url_loader(source)
    elif type == 'zip':
        docs = universal_zip_loader(source)
    return docs
    