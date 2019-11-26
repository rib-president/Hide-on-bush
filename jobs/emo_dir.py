#!/usr/bin/env python
import argparse
import glob
import os
from collections import OrderedDict
from pprint import pformat

__author__ = 'Chris Rands'
__copyright__ = 'Copyright (c) 2017-2018, Chris Rands'

try:
    range = xrange
except NameError:
    pass  # Python 3

EMOTICONS = [':)', ':D', ':P', ':S', ':(', '=)', '=/', ':/', ':{', ';)']
# TODO: Add other alphabets as options including real emojis
MAX_STR_LEN = 70

def chunk_string(in_s, n):
    """Chunk string to max length of n"""
    return '\n'.join('{}\\'.format(in_s[i:i+n]) for i in range(0, len(in_s), n)).rstrip('\\')


def encode_string(in_s, alphabet):
    """Convert input string to encoded output string with the given alphabet"""
    # Using OrderedDict to guarantee output order is the same
    # Note Python 2 and 3 inputs differ slightly due to pformat()
    d1 = OrderedDict(enumerate(alphabet))
    d2 = OrderedDict((v, k) for k, v in d1.items())
    return ('from collections import OrderedDict\n'
            'exec("".join(map(chr,[int("".join(str({}[i]) for i in x.split())) for x in\n'
            '"{}"\n.split("  ")])))\n'.format(pformat(d2), chunk_string('  '.join(
            ' '.join(d1[int(i)] for i in str(ord(c))) for c in in_s), MAX_STR_LEN)))


def main(in_file, out_file):
    with open(in_file) as in_f, open(out_file, 'w') as out_f:
        # This assumes it's ok to read the entire input file into memory
        out_f.write(encode_string(in_f.read(), EMOTICONS))

if __name__ == '__main__':
    path = '/home/rooney/code_piece/import_test/'
    output_path = '/home/rooney/'
    filelist = sorted(os.listdir(path))
    pylist = []
    print filelist
    for pyfile in filelist:
        if pyfile.find('.py') is not -1:
            pylist.append(pyfile)
    #pyfile = path.split('/')[-1]
    #pylist.append(pyfile)
            
    
    for pyfile in pylist:
        outname = 'emo_' + pyfile
        contents = open(path + '/' + pyfile,'r').read()
        #contents = open(path,'r').read
        #print contents
        open(output_path + outname, 'w').write(contents)
        
        input_py = path + pyfile
        output_py = path + outname
        main(input_py, output_py)
    
    
