import pickle
from tqdm import tqdm

from configs import BaseConfigs
from instance import ACSAInstance, ATSAInstance
import xml.etree.ElementTree

class Preprocessor(object):
    def __init__(self,configs):
        self.configs=configs

    def preprocess_ACSA(self,src,dst):
        instances=[]
        with open(src,'r',encoding='utf8') as f:
            tree=xml.etree.ElementTree.parse(f)
            for node in tqdm(tree.iterfind('sentence')):
                sentence=node.findtext('text')

                for category_node in node.iterfind('aspectCategories/aspectCategory'):
                    instance = ACSAInstance()
                    instance.sentence=sentence
                    instance.category=ACSAInstance.category2index[category_node.get('category')]
                    instance.polarity=ACSAInstance.polarity2index[category_node.get('polarity')]

                    instances.append(instance)

        with open(dst,'wb') as f:
            pickle.dump(instances,f)

    def preprocess_ATSA(self,src,dst):
        instances=[]
        with open(src,'r',encoding='utf8') as f:
            tree=xml.etree.ElementTree.parse(f)
            for node in tqdm(tree.iterfind('sentence')):
                sentence=node.findtext('text')

                for term_node in node.iterfind('aspectTerms/aspectTerm'):
                    instance = ATSAInstance()
                    instance.sentence=sentence
                    instance.aspect_term=term_node.get('term')
                    instance.aspect_from_pos=int(term_node.get('from'))
                    instance.polarity=ATSAInstance.polarity2index[term_node.get('polarity')]

                    instances.append(instance)

        with open(dst,'wb') as f:
            pickle.dump(instances,f)

if __name__=='__main__':
    configs=BaseConfigs()
    processor=Preprocessor(configs)

    processor.preprocess_ACSA('data/ACSA/train.xml','data/ACSA/train.pickle')
    processor.preprocess_ACSA('data/ACSA/dev.xml','data/ACSA/dev.pickle')
    processor.preprocess_ACSA('data/ACSA/test.xml', 'data/ACSA/test.pickle')

    processor.preprocess_ATSA('data/ATSA/train.xml','data/ATSA/train.pickle')
    processor.preprocess_ATSA('data/ATSA/dev.xml','data/ATSA/dev.pickle')
    processor.preprocess_ATSA('data/ATSA/test.xml', 'data/ATSA/test.pickle')
