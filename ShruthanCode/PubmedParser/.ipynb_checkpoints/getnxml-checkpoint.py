import os
import requests

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree

import tarfile

def get_nxml_from_pmcid(pmcid):
    if pmcid.startswith("PMC"):
        pmcid = pmcid[3:]
    query_string = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}"
    resp = requests.get(query_string)
    
    if resp.status_code == 200:
        tree = ElementTree(ET.fromstring(resp.content))

        download_link = None
        root = tree.getroot()
        for item in root.findall('./records'):
            for record in item:
                for link in record:
                    if link.attrib['format'] == 'tgz':
                        download_link = "https://" + link.attrib['href'][6:] #remove file://, which is 6 characters
                        break

        response = requests.get(download_link, stream=True)
        target_path = f"tars/PMC{pmcid}.tar.gz"
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        else:
            print(f"ERROR: {response.status_code}")

        file = tarfile.open(target_path)
        file.extractall('./nxmls')
        file.close()
        
        curr_pmc_dir = f"./nxmls/PMC{pmcid}"
        for file in os.listdir(curr_pmc_dir):
            if file.endswith('.nxml'):
                return os.path.join(curr_pmc_dir, file)
    else:
        print(f"ERROR: {resp.status_code}")