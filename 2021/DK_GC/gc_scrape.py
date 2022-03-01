import pandas as pd
from bs4 import BeautifulSoup
import codecs


html_doc = codecs.open("DKPGA/DK_GC/gc_html.html", 'r')

soup = BeautifulSoup(html_doc, 'html.parser')

names = soup.find_all("span", {"class": "_2gKYvnBDeTo-MfsrhVznG1"})
for name in names:
    print(name.text)