import os
import pandas as pd


def pdf_to_excel(output_path, sheetnames, pdfs, name='clustering.xlsx'):
    if isinstance(sheetnames, str):
        sheetnames = [sheetnames]
        pdfs = [pdfs]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fp = output_path + '/' + name
    writer = pd.ExcelWriter(fp, engine='xlsxwriter')
    for tab, pdf in zip(sheetnames, pdfs):
        pdf.to_excel(writer, tab)
    writer.save()
    writer.close()


def pdf_to_csv(output_path, names, pdfs):
    if isinstance(names, str):
        names = [names]
        pdfs = [pdfs]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for tab, pdf in zip(names, pdfs):
        fp = output_path + '/' + tab + '.csv'
        pdf.to_csv(fp)
