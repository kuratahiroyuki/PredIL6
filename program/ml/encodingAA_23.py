import pandas as pd
import numpy as np
import re, math
from collections import Counter

def AAC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    '''
    #header = ['#', 'label']
    header = []  
    for i in AA:
        header.append(i)   
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1] ###
        sequence = re.sub('X', '', sequence) # addition
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        #code = [sequence, label]
        code = [] ###
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[0]):
            minLen = len(i[0])
    return minLen

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair

def CTDT(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0],  i[1] #
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        encodings.append(code)
    return encodings


def CKSAAGP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p + '.gap' + str(g))
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0],i[1]
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                        sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)
        encodings.append(code)
    return encodings
    

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[0]):
            minLen = len(i[0])
    return minLen

def CKSAAP(fastas, gap=0, **kw): #revision gap=5 default
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA='ARNDCQEGHILKMFPSTWYVX'
    
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    '''
    #header = ['#', 'label']
    header =[]
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings


def get_min_sequence_length_1(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(re.sub('X', '', i[0])):
            minLen = len(re.sub('X', '', i[0]))
    return minLen

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(fastas, lambdaValue=5, w=0.05, **kw):# adjusted

    if get_min_sequence_length_1(fastas) < lambdaValue + 1:
        print(
            'Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
        return 0

    dataFile = './PAAC.txt' # adjusted
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    
    encodings=[]
    '''
    #header = ['#', 'label']
    header = []
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    encodings.append(header)
    '''
    
    for i in fastas:
        sequence, label = i[0], i[1]
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
        
    return encodings

def check_fasta_with_equal_length(fastas):
    status = True
    lenList = set()
    for i in fastas:
        lenList.add(len(i[0]))
    if len(lenList) == 1:
        return True
    else:
        return False
        
def AAINDEX(fastas, props=None, **kw):
    if check_fasta_with_equal_length(fastas) == False:
        print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'

    #fileAAindex = re.sub('descproteins$', '', os.path.split(os.path.realpath(__file__))[
    #    0]) + r'\data\AAindex.txt' if platform.system() == 'Windows' else re.sub('descproteins$', '', os.path.split(
    #    os.path.realpath(__file__))[0]) + r'/data/AAindex.txt'
    #fileAAindex = data_path + '/descproteins/data/AAindex.txt'
    fileAAindex = './AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    #  use the user inputed properties
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex
    
    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for pos in range(1, len(fastas[0][0]) + 1):
        for idName in AAindexName:
            header.append('SeqPos.' + str(pos) + '.' + idName)
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        #code = [name, label]
        code = []
        for aa in sequence:
            if aa == 'X':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encodings.append(code)
    return encodings

def BLOSUM62(fastas, **kw):

    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }
    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for i in range(1, len(fastas[0][0]) * 20 + 1):
        header.append('blosum62.F'+str(i))
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        #code = [name, label]
        code = []
        for aa in sequence:
            code = code + blosum62[aa]
        encodings.append(code)
    return encodings


def ZSCALE(fastas, **kw):

    zscale = {
        'A': [0.24,  -2.32,  0.60, -0.14,  1.30], # A
        'C': [0.84,  -1.67,  3.71,  0.18, -2.65], # C
        'D': [3.98,   0.93,  1.93, -2.46,  0.75], # D
        'E': [3.11,   0.26, -0.11, -0.34, -0.25], # E
        'F': [-4.22,  1.94,  1.06,  0.54, -0.62], # F
        'G': [2.05,  -4.06,  0.36, -0.82, -0.38], # G
        'H': [2.47,   1.95,  0.26,  3.90,  0.09], # H
        'I': [-3.89, -1.73, -1.71, -0.84,  0.26], # I
        'K': [2.29,   0.89, -2.49,  1.49,  0.31], # K
        'L': [-4.28, -1.30, -1.49, -0.72,  0.84], # L
        'M': [-2.85, -0.22,  0.47,  1.94, -0.98], # M
        'N': [3.05,   1.62,  1.04, -1.15,  1.61], # N
        'P': [-1.66,  0.27,  1.84,  0.70,  2.00], # P
        'Q': [1.75,   0.50, -1.44, -1.34,  0.66], # Q
        'R': [3.52,   2.50, -3.50,  1.99, -0.17], # R
        'S': [2.39,  -1.07,  1.15, -1.39,  0.67], # S
        'T': [0.75,  -2.18, -1.12, -1.46, -0.40], # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02], # V
        'W': [-4.36,  3.94,  0.59,  3.44, -1.59], # W
        'Y': [-2.54,  2.44,  0.43,  0.04, -1.47], # Y
        'X': [0.00,   0.00,  0.00,  0.00,  0.00], # -  ###
    }
    encodings = []
    """
    header = ['#', 'label']
    for p in range(1, len(fastas[0][1])+1):
        for z in ('1', '2', '3', '4', '5'):
            header.append('Pos'+str(p) + '.ZSCALE' + z)
    encodings.append(header)
    """

    for i in fastas:
        sequence, label = i[0], i[1]
        code = []
        for aa in sequence:
            code = code + zscale[aa]
        encodings.append(code)
    return encodings


def TPC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    triPeptides = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    #header = ['#', 'label'] + triPeptides
    #encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label = re.sub('X', '', i[0]), i[1]

        code = []
        tmpCode = [0] * 8000
        for j in range(len(sequence) - 3 + 1):
            tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j+1]]*20 + AADict[sequence[j+2]]] = tmpCode[AADict[sequence[j]] * 400 + AADict[sequence[j+1]]*20 + AADict[sequence[j+2]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings


def DPC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    triPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    #header = ['#', 'label'] + triPeptides
    #encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label = re.sub('X', '', i[0]), i[1]

        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]]*20 + AADict[sequence[j+1]]] = tmpCode[ AADict[sequence[j]]*20 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return encodings


def CalculateKSCTriad(sequence, gap, features, AADict):
    res = []
    for g in range(gap + 1):
        myDict = {}
        for f in features:
            myDict[f] = 0

        for i in range(len(sequence)):
            if i + g + 1 < len(sequence) and i + 2 * g + 2 < len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i + g + 1]] + '.' + AADict[
                    sequence[i + 2 * g + 2]]
                myDict[fea] = myDict[fea] + 1

        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            res.append((myDict[f] - minValue) / maxValue)

    return res

def CTriad(fastas, gap = 0, **kw):
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }

    myGroups = sorted(AAGroup.keys())

    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g

    features = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]
    encodings = []
    """
    header = ['#', 'label']
    for f in features:
        header.append(f)
    encodings.append(header)
    """
    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label =  re.sub('X', '', i[0]), i[1]
        code = []
        if len(sequence) < 3:
            print('Error: for "CTriad" encoding, the input fasta sequences should be greater than 3. \n\n')
            return 0
        code = code + CalculateKSCTriad(sequence, 0, features, AADict)
        encodings.append(code)

    return encodings


def GAAC(fastas, **kw):
    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharge': 'STCPNQ'
    }

    groupKey = group.keys()

    encodings = []
    """
    header = ['#', 'label']
    for key in groupKey:
        header.append(key)
    encodings.append(header)
    """
    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label =  re.sub('X', '', i[0]), i[1]
        code = []
        count = Counter(sequence)
        myDict = {}
        for key in groupKey:
            for aa in group[key]:
                myDict[key] = myDict.get(key, 0) + count[aa]

        for key in groupKey:
            code.append(myDict[key]/len(sequence))
        encodings.append(code)

    return encodings


def GDPC(fastas, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    baseNum = len(groupKey)
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    """
    header = ['#', 'label'] + dipeptide
    encodings.append(header)
    """
    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label =  re.sub('X', '', i[0]), i[1]
        code = []
        myDict = {}
        for t in dipeptide:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 2 + 1):
            myDict[index[sequence[j]] + '.' + index[sequence[j + 1]]] = myDict[index[sequence[j]] + '.' + index[
                sequence[j + 1]]] + 1
            sum = sum + 1

        if sum == 0:
            for t in dipeptide:
                code.append(0)
        else:
            for t in dipeptide:
                code.append(myDict[t] / sum)
        encodings.append(code)

    return encodings


def GTPC(fastas, **kw):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKey = group.keys()
    baseNum = len(groupKey)
    triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    encodings = []
    #header = ['#', 'label'] + triple
    #encodings.append(header)

    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label =  re.sub('X', '', i[0]), i[1]
        code = []
        myDict = {}
        for t in triple:
            myDict[t] = 0

        sum = 0
        for j in range(len(sequence) - 3 + 1):
            myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] + 1
            sum = sum +1

        if sum == 0:
            for t in triple:
                code.append(0)
        else:
            for t in triple:
                code.append(myDict[t]/sum)
        encodings.append(code)

    return encodings


def Count_CTDC(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum


def CTDC(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    #header = ['#', 'label']
    #for p in property:
    #    for g in range(1, len(groups) + 1):
    #        header.append(p + '.G' + str(g))
    #encodings.append(header)
    
    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label =  re.sub('X', '', i[0]), i[1] 
        code = []
        for p in property:
            c1 = Count_CTDC(group1[p], sequence) / len(sequence)
            c2 = Count_CTDC(group2[p], sequence) / len(sequence)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        encodings.append(code)
    return encodings

def Count_CTDD(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code


def CTDD(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')


    encodings = []
    #header = ['#', 'label']
    #for p in property:
    #    for g in ('1', '2', '3'):
    #        for d in ['0', '25', '50', '75', '100']:
    #            header.append(p + '.' + g + '.residue' + d)
    #encodings.append(header)

    for i in fastas:
        #name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
        #code = [name, label]
        sequence, label =  re.sub('X', '', i[0]), i[1] 
        code = []
        for p in property:
            code = code + Count_CTDD(group1[p], sequence) + Count_CTDD(group2[p], sequence) + Count_CTDD(group3[p], sequence)
        encodings.append(code)
    return encodings

    
def EAAC(fastas, window=5, **kw):
    """
    if check_sequences.check_fasta_with_equal_length == False:
        print('Error: for "EAAC" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0

    if window < 1:
        print('Error: the sliding window should be greater than zero' + '\n\n')
        return 0

    if check_sequences.get_min_sequence_length(fastas) < window:
        print('Error: all the sequence length should be larger than the sliding window :' + str(window) + '\n\n')
        return 0
    """
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    #header = ['#', 'label']
    #for w in range(1, len(fastas[0][1]) - window + 2):
    #    for aa in AA:
    #        header.append('SW.'+str(w)+'.'+aa)
    #encodings.append(header)

    for i in fastas:
        #name, sequence, label = i[0], i[1], i[2]
        #code = [name, label]
        sequence, label = i[0], i[1]
        #sequence = re.sub('X', '', sequence) # addition        
        code = []
        for j in range(len(sequence)):
            if j < len(sequence) and j + window <= len(sequence):
                count = Counter(sequence[j:j+window])
                for key in count:
                    count[key] = count[key] / len(sequence[j:j+window])
                """
                for aa in AA:
                    code.append(count[aa])
                for aa in 'X':
                    code.append(0.0) 
                """
                                    
                for aa in AA+'X':
                    if aa == 'X':
                        code.append(0.0) 
                    else:  
                        code.append(count[aa])                                  
        encodings.append(code)
    return encodings
