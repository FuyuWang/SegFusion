import logging
import collections
import copy
import logging
import os
import numpy as np
import pathlib
import re
import xml.etree.ElementTree as ET


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture everything


# Output file names.
out_prefix = "timeloop-model."
report_prefix = out_prefix + 'stats.txt'
xml_file_name = out_prefix + "map+stats.xml"

def run_config(filename):
    filename = pathlib.Path(filename).resolve()
    report_file = filename / report_prefix
    status_dict = dict()
    if report_file.exists():
        level_latency = []
        used_bandwidth = 0
        with open(report_file, 'r') as f:
            lines = f.readlines()
        flag = False
        for line in lines:
            m = re.match(r"Energy: (.*) uJ", line)
            # m = re.match(r"Total topology energy: (.*) pJ", line)
            if m:
                energy = m.group(1)
                status_dict['energy'] = float(energy)
            else:
                # m = re.match(r"Max topology cycles: (.*)", line)
                m = re.match(r"Cycles: (.*)", line)
                if m:
                    cycle = m.group(1)
                    status_dict['cycles'] = int(cycle)
    return status_dict
