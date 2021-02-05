from enum import Enum


class DSTitle(Enum):
    """
    Titles of supported datasets
    """
    rPPG = 'rPPG'
    Mahnob = 'Mahnob-HCI'
    UBFC = 'UBFC-RPPG'
    VIPL = 'VIPL-HR'
    # VIPLv2 = 'VIPL-HR_V2'  # currently is not implemented
    DEAP = 'DEAP'
    RePSS_Train = 'RePSS_Train'
    RePSS_Test = 'RePSS_Test'
    DCC_SFEDU = 'DCC-SFEDU'
    SFEDU_2014 = 'SFEDU-2014'
    # test = 'test'
