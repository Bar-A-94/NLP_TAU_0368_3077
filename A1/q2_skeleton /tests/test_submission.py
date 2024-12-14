import inspect
import itertools
import os
import glob
import pathlib
import pytest


def test_pdf():
    #make sure to title the pdf file name as id1_id2_id3.pdf i.e: 123456789_123456789_123456789.pdf
    pdf_list = list(glob.glob("*.pdf"))
    assert len(pdf_list)<2,"Only submit a single pdf file"
    assert len(pdf_list)>0,"Only submit answers in a pdf format"


    pdf_path = pathlib.Path(pdf_list[0])

    for id_el in pdf_path.name[:-4].split("_"):
        print("id:",id_el)
        assert len(id_el)==9,"ID number is in a wrong format"
        
def test_npy():
    path = pathlib.Path("saved_params_40000.npy")
    assert path.exists()
