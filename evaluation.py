from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import subprocess


def bleu_score(ref_file, hyp_file):
    """Computes corpus level BLEU score with Moses' multi-bleu.pl script

    Arguments:
        ref_file (str): Path to the reference file
        hyp_file (str): Path to the hypothesis file

    Returns:
        tuple: Tuple (BLEU, details) containing the bleu score and the detailed output of the perl script

    Raises:
        ValueError: Raises error if the perl script fails for some reason
    """
    command = 'perl scripts/multi-bleu.pl ' + ref_file + ' < '+hyp_file
    c = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    details, error = c.communicate()
    if not details.startswith('BLEU ='):
        raise ValueError('Error in BLEU score computation:\n' + error)
    else:
        BLEU_str = details.split(' ')[2][:-1]
        BLEU = float(BLEU_str)
        return BLEU, details
