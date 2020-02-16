import os
import json
import string
import pickle
import numpy as np

def parse_snips_original(path):

    a = os.listdir(".")

    ds_types = {"lights": [], "music": []}
    a_t_synonm = {}
    synom_t_a = []


    adds_slots = {}
    #
    # intents =

    folders = [elem for elem in os.listdir(path) if not 'fr' in elem and not ".DS_Store" in elem and "smart" in elem]

    for num, elem in enumerate(folders):
        json_files = [el for el in os.listdir(os.path.join(path, elem)) if el.endswith("dataset.json")]

        for dataset in json_files:

            print(f"Processing {dataset}")
            with open(os.path.join(path, elem, dataset), 'r') as file:

                json_f = json.load(file)

                additional_entities = json_f['entities']
                for k, v in additional_entities.items():

                    if 'data' in v.keys():
                        syns = v["data"]

                        for i in syns:

                            if not i["synonyms"]:
                                if i["value"]:

                                    if not k in adds_slots.keys():

                                        adds_slots[k] = [i["value"]]
                                    else:
                                        adds_slots[k].append(i["value"])
                            else:

                                if i["value"]:

                                    if not i["value"] in a_t_synonm.keys():
                                        a_t_synonm[i["value"]] = i["synonyms"]
                                    else:
                                        a_t_synonm[i["value"]].extend(i["synonyms"])

                                    for j in i["synonyms"]:
                                        synom_t_a.append((j, i["value"]))


                ints = []
                seqin = []
                seqout = []

                intents = json_f["intents"]
                for k, v in intents.items():

                    v = v['utterances']
                    for i in v:
                        slot_data = i["data"]
                        full_text = []
                        seqout_cur = []

                        for j in slot_data:
                            val = j["text"].lower().translate(str.maketrans("","", string.punctuation))
                            if val.endswith(" "):
                                val = val[:-1]
                            if val.startswith(" "):
                                val = val[1:]
                            val = val.split(" ")
                            if "entity" in j.keys():

                                if 'snips/' in j["entity"]:
                                    aa = j["entity"].strip("snips/")
                                else:
                                    aa = j["entity"]
                                if aa.endswith("EN"):
                                    aa = aa[:-2]

                                if aa.endswith("1-100_"):
                                    aa = aa[:-6]
                                seqout_cur.append("B-"+aa)
                                for m in range(len(val)-1):
                                    seqout_cur.append("I-"+aa)
                            else:
                                for m in range(len(val)):
                                    seqout_cur.append("O")

                            full_text.extend(val)


                        cur_text = " ".join(full_text)
                        cur_seqout = " ".join(seqout_cur)
                        seqout.append(cur_seqout+"\n")
                        seqin.append(cur_text+"\n")
                        ints.append(k+"\n")

            dataset_name = elem
            os.makedirs(f"parsed_snips_orgin/{dataset_name}{num}", exist_ok=True)
            with open (f"parsed_snips_orgin/{dataset_name}{num}/label", "w") as f1:
                f1.writelines(ints)

            with open (f"parsed_snips_orgin/{dataset_name}{num}/seq.in", "w") as f1:
                f1.writelines(seqin)

            with open (f"parsed_snips_orgin/{dataset_name}{num}/seq.out", "w") as f1:
                f1.writelines(seqout)





                # for k, v in intents.items():
    with open("sta.pickle", "wb") as f:
        pickle.dump(synom_t_a, f)

    with open("ats.pickle", "wb") as f:
        pickle.dump(a_t_synonm, f)

    with open("adds_slot.pickle", "wb") as f:
        pickle.dump(adds_slots, f)



def categorize_snips(path):


    folders = [elem for elem in os.listdir(path) if not ".DS_Store" in elem and not elem.endswith(".md")]
    cats = {}
    for folder in folders:
        train_text = 'train_'+folder+"_full.json"
        vaild_text = "validate_"+folder+".json"

        with open(os.path.join(path, folder, train_text), 'r', encoding='latin-1') as f:
            train = json.load(f)

        with open(os.path.join(path, folder, vaild_text), 'r', encoding='latin-1') as f:
            val = json.load(f)

        text = []

        data = train[folder]
        for elem in data:
            data_txt = elem["data"]
            sent = ''
            for el in data_txt:
                cur_txt = el['text'].lower().translate(str.maketrans("","", string.punctuation))
                sent +=cur_txt
            text.append(sent)
        # cats[folder] = text

        data = val[folder]
        for elem in data:
            data_txt = elem["data"]
            sent = ''
            for el in data_txt:
                cur_txt = el['text'].lower().translate(str.maketrans("", "", string.punctuation))
                sent += cur_txt
            text.append(sent)
        cats[folder] = text

    new_folders = [elem for elem in os.listdir("parsed_snips_orgin")]

    for fl in new_folders:
        text_new = []
        with open(os.path.join('parsed_snips_orgin', fl, 'seq.in'), 'r') as f:
            text_new = f.readlines()

        for i in range(len(text_new)):
            text_new[i] = text_new[i][:-1]

        new_cat = fl.split("-")[1]
        if new_cat not in cats.keys():
            cats[new_cat] = text_new
        else:
            cats[new_cat].extend(text_new)

    with open("categories_info.pickle", 'wb') as f:

        pickle.dump(cats, f)


def unite_datasets(path, train = 0.6, test = 0.2, valid = 0.2):

    assert train+test+valid == 1

    folders = [elem for elem in os.listdir(os.path.join(path, 'parsed_snips_orgin')) if not ".DS_Store" in elem and not elem.endswith(".md")]

    for folder in folders:

        with open(os.path.join(path, 'parsed_snips_orgin', folder, 'label'), 'r') as f:

            intent = f.readlines()

        with open(os.path.join(path, 'parsed_snips_orgin',folder, 'seq.in'), 'r') as f:
            text = f.readlines()

        with open(os.path.join(path,'parsed_snips_orgin', folder, 'seq.out'), 'r') as f:
            slots = f.readlines()
        #
        # intent = [elem[:-1] for elem in intent]
        # text = [elem[:-1] for elem in text]
        # slots= [elem[:-1] for elem in slots]

        idxs = np.arange(0, len(intent))

        np.random.shuffle(idxs)

        train_size = int(len(intent)*train)
        valid_size = int(len(intent)*valid)
        test_size = len(intent) - train_size - valid_size


        tr_idxs = idxs[:train_size]
        val_idxs = idxs[train_size:train_size+valid_size]
        test_idxs = idxs[train_size+valid_size:train_size+valid_size+test_size]


        train_intent = [ intent[i] for i in tr_idxs ]
        valid_intent = [ intent[i] for i in val_idxs ]
        test_intent = [ intent[i] for i in test_idxs ]

        train_text = [text[i] for i in tr_idxs]
        valid_text = [text[i] for i in val_idxs]
        test_text = [text[i] for i in test_idxs]

        train_slot = [slots[i] for i in tr_idxs]
        valid_slot = [slots[i] for i in val_idxs]
        test_slot = [slots[i] for i in test_idxs]

        with open(os.path.join(path, 'snips/train/label'), 'a') as f:
            f.writelines(train_intent)

        with open(os.path.join(path, 'snips/train/seq.in'), 'a') as f:
            f.writelines(train_text)

        with open(os.path.join(path, 'snips/train/seq.out'), 'a') as f:
            f.writelines(train_slot)


        with open(os.path.join(path, 'snips/valid/label'), 'a') as f:
            f.writelines(valid_intent)

        with open(os.path.join(path, 'snips/valid/seq.in'), 'a') as f:
            f.writelines(valid_text)

        with open(os.path.join(path, 'snips/valid/seq.out'), 'a') as f:
            f.writelines(valid_slot)



        with open(os.path.join(path, 'snips/test/label'), 'a') as f:
            f.writelines(test_intent)

        with open(os.path.join(path, 'snips/test/seq.in'), 'a') as f:
            f.writelines(test_text)

        with open(os.path.join(path, 'snips/test/seq.out'), 'a') as f:
            f.writelines(test_slot)




# parse_snips_original("../data/snips_slu_data_v1.0")
unite_datasets("../data")
# categorize_snips("../data/2017-06-custom-intent-engines")










