import copy
import random
from instance import ATSAInstance, ACSAInstance


def merge_instances(instances):
    instances_by_sentence = {}
    for x in instances:
        if instances_by_sentence.get(x.sentence) is None:
            instances_by_sentence[x.sentence] = []
        instances_by_sentence[x.sentence].append(x)

    merged_instances=[]
    for sentence_pack in instances_by_sentence.values():
        instance=ATSAInstance()
        instance.sentence=sentence_pack[0].sentence
        instance.word_tokens=sentence_pack[0].word_tokens
        instance.aspect_term=[]
        instance.polarity=[]
        instance.aspect_from_pos=[]
        instance.aspect_from=[]
        instance.aspect_to=[]

        sentence_pack=sorted(sentence_pack,key=lambda x:x.aspect_from_pos)
        for x in sentence_pack:
            instance.aspect_term.append(x.aspect_term)
            instance.polarity.append(x.polarity)
            instance.aspect_from_pos.append(x.aspect_from_pos)

        merged_instances.append(instance)
    return merged_instances

def merge_instances_ACSA(instances):
    instances_by_sentence = {}
    for x in instances:
        if instances_by_sentence.get(x.sentence) is None:
            instances_by_sentence[x.sentence] = []
        instances_by_sentence[x.sentence].append(x)

    merged_instances=[]
    for sentence_pack in instances_by_sentence.values():
        instance=ACSAInstance()
        instance.sentence=sentence_pack[0].sentence
        instance.category=[]
        instance.polarity=[]

        random.shuffle(sentence_pack)

        for x in sentence_pack:
            instance.category.append(x.category)
            instance.polarity.append(x.polarity)
        merged_instances.append(instance)
    return merged_instances

def merge_instances_dummy(instances):
    instances=copy.deepcopy(instances)
    for x in instances:
        x.aspect_term=[x.aspect_term]
        x.aspect_from_pos=[x.aspect_from_pos]
        x.polarity=[x.polarity]
    return instances

def merge_instances_dummy_ACSA(instances):
    instances=copy.deepcopy(instances)
    for x in instances:
        x.category=[x.category]
        x.polarity=[x.polarity]
    return instances
