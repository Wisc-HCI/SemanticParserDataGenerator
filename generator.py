
'''
//%[while_hum_answer]('training':'800')
//    pass
//~[while_hum_answer]
//    while ~[hum] ~[answer_bare]
//    ~[answer_bare] until ~[hum]
//    ~[answer_bare] while ~[hum]
//    continue ~[answering] until ~[hum]
//    continue ~[answering] while ~[hum]
//    continue ~[answering]
//    loop on ~[answering]
//    loop on ~[answering] until ~[hum]
//    do not stop ~[answering] until ~[hum]
//    do not stop ~[answering]
'''

import json
import numpy as np

class Generator():

    def __init__(self):
        self.training_data = []
        self.action_leaves = {"ask": {"starting": [], "bare": [], "present": []},
                              "answer": {"starting": [], "bare": [], "present": []},
                              "say": {"starting": [], "bare": [], "present": []},
                              "place": {"starting": [], "bare": [], "present": []},
                              "grab": {"starting": [], "bare": [], "present": []},
                              "handoff": {"starting": [], "bare": [], "present": []},
                              "listen": {"starting": [], "bare": [], "present": []},
                              "gaze": {"starting": [], "bare": [], "present": []},
                              "move": {"starting": [], "bare": [], "present": []},
                              "gesture": {"starting": [], "bare": [], "present": []}
                      }
        self.action_leaves_behavior = {}
        self.state_leaves = {"human": [], "robot": [], "env": []}
        self.primitives = {"ask": ["speech"],
                           "answer": ["speech"],
                           "say": ["speech"],
                           "place": ["item","location"],
                           "grab": ["item"],
                           "handoff": ["item"],
                           "listen": ["null"],
                           "gaze": ["gaze"],
                           "move": ["location"],
                           "gesture": ["gesture"],
                           "human": ["hum"],
                           "env": ["ambig"],
                           "robot": ["rob"]
                      }

        # set up probabilities of a particular example being generated
        self.construct_probs = [
                                0.1,    # action
                                0.1,    # not
                                0.1,    # and
                                0.1,    # or
                                0.2,    # while
                                0.2,    # if
                                0.2,    # step
                               ]

        self.constructs = ["action","not","and","or","while","if","step"]

        # continuation probabilities
        self.nested_while_prob = 0.3
        self.nested_step_prob = 0.3
        self.nested_if_prob = 0.3

        # cap on how many nests
        self.nested_cap = 2

    def load(self, json_file):
        with open(json_file, "rb") as infile:
            leaves_dicts = json.load(infile)["rasa_nlu_data"]["common_examples"]
        for leaf_dict in leaves_dicts:
            text = leaf_dict["text"]
            intent = leaf_dict["intent"]

            if intent in self.state_leaves:
                self.state_leaves[intent].append(text)

            else:
                for action in self.action_leaves:
                    if action in intent:
                        if "present" in intent:
                            self.action_leaves[action]["present"].append(text)
                        elif "bare" in intent:
                            self.action_leaves[action]["bare"].append(text)
                        else:
                            self.action_leaves[action]["starting"].append(text)
                        break
        self.action_leaves_behavior["gaze"] = self.action_leaves["gaze"]
        self.action_leaves_behavior["gesture"] = self.action_leaves["gesture"]

    def generate(self, n):
        training_examples = []
        i = 0
        while i < n:
            training_example = {"name": None, "string": None, "args": None}  # {"name": STRING, "string": Y, "args": [DICT]}
            self.fill_training_example(training_example, 0)
            if training_example["name"] is not None:
                training_example = self.write_out_training_example(training_example)

                if training_example not in training_examples:
                    training_examples.append(training_example)
                    i += 1

        with open("generated_data.txt","w") as outfile:
            for tx in training_examples:
                outfile.write("{}\n".format(tx))

    def fill_training_example(self, example, depth, get_state=False, handling_front=True, handling_continuing=False, available_choices=None):
        if not get_state:
            if available_choices is not None:
                choice = np.random.choice(available_choices)
                if depth == 2:
                    choice = "action"
            elif depth == 2:
                choice = "action"
            else:
                choice = np.random.choice(self.constructs,p=self.construct_probs)

            if choice == "action":
                choice = np.random.choice(list(self.action_leaves.keys()))
            elif choice == "beh_action":
                choice = np.random.choice(list(self.action_leaves_behavior.keys()))
        else:
            choice = np.random.choice(list(self.state_leaves.keys()))
        name = choice.upper()

        # leaf
        if choice in self.action_leaves:
            example["name"] = name
            if handling_front:
                text = np.random.choice(self.action_leaves[choice]["starting"])
            elif handling_continuing:
                text = np.random.choice(self.action_leaves[choice]["present"])
            else:
                text = np.random.choice(self.action_leaves[choice]["bare"])
            args = self.primitives[choice]
            example["string"] = text
            example["args"] = args

        if choice == "human" or choice == "robot" or choice == "env":
            example["name"] = name
            text = np.random.choice(self.state_leaves[choice])
            args = self.primitives[choice]
            example["string"] = text
            example["args"] = args

        # step
        elif choice == "step":
            example["name"] = name
            example["args"] = []
            first_arg = {"name": None, "string": None, "args": None}
            second_arg = {"name": None, "string": None, "args": None}
            if handling_front:
                self.fill_training_example(first_arg, depth+1, handling_front=True)
            else:
                self.fill_training_example(first_arg, depth+1, handling_front=False)
            self.fill_training_example(second_arg, depth+1, handling_front=False)
            example["args"].append(first_arg)
            example["args"].append(second_arg)
            example["string"] = np.random.choice(["11 and then 22",
                                                  "11 and next 22",
                                                  "11 and then next 22",
                                                  "11 next 22",
                                                  "11 then 22",
                                                  "11 and following that 22"])

        # if
        elif choice == "if":
            example["name"] = name
            example["args"] = []
            first_arg = {"name": None, "string": None, "args": None}
            second_arg = {"name": None, "string": None, "args": None}
            self.fill_training_example(first_arg, depth+1, get_state=True)
            self.fill_training_example(second_arg, depth+1, handling_front=True)
            example["args"].append(first_arg)
            example["args"].append(second_arg)
            example["string"] = np.random.choice(["if 11 then 22",
                                                  "only if 11 then 22",
                                                  "22 if 11",
                                                  "22 only if 11"])

        # while
        if choice == "while":
            example["name"] = name
            example["args"] = []
            first_arg = {"name": None, "string": None, "args": None}
            second_arg = {"name": None, "string": None, "args": None}
            self.fill_training_example(first_arg, depth+1, get_state=True)

            # decide if to talk in present tense
            talk_in_present = np.random.choice(["present","not_present"])
            if talk_in_present:
                self.fill_training_example(second_arg, depth+1, handling_continuing=True, available_choices=["action","step","or","and"])
                example["args"].append(first_arg)
                example["args"].append(second_arg)
                example["string"] = np.random.choice(["while 11 22",
                                                      "22 until 11",
                                                      "22 while 11"
                                                     ])
            else:
                self.fill_training_example(second_arg, depth+1)
                example["args"].append(first_arg)
                example["args"].append(second_arg)
                example["string"] = np.random.choice(["continue 22 until 11",
                                                      "continue 22 while 11"
                                                     ])

        # not
        elif choice == "not":
            example["name"] = name
            example["args"] = []
            first_arg = {"name": None, "string": None, "args": None}
            self.fill_training_example(first_arg, depth+1, handling_front=False, available_choices=["or","and","step","action"])
            example["args"].append(first_arg)
            example["string"] = np.random.choice(["do not 11"])


        # or
        elif choice == "or":
            example["name"] = name
            example["args"] = []
            first_arg = {"name": None, "string": None, "args": None}
            second_arg = {"name": None, "string": None, "args": None}
            if handling_front:
                self.fill_training_example(first_arg, depth+1, handling_front=True)
            else:
                self.fill_training_example(first_arg, depth+1, handling_front=False)
            self.fill_training_example(second_arg, depth+1, handling_front=False)
            example["args"].append(first_arg)
            example["args"].append(second_arg)
            example["string"] = "11 or 22"

        # and
        elif choice == "and":
            example["name"] = name
            example["args"] = []
            first_arg = {"name": None, "string": None, "args": None}
            second_arg = {"name": None, "string": None, "args": None}

            where_behavior = np.random.choice(["front", "back"])
            if where_behavior == "front":
                if handling_front:
                    self.fill_training_example(first_arg, depth+1, handling_front=True, available_choices=["beh_action"])
                else:
                    self.fill_training_example(first_arg, depth+1, handling_front=False, available_choices=["beh_action"])
                self.fill_training_example(second_arg, depth+1, handling_front=False)
            else:
                if handling_front:
                    self.fill_training_example(first_arg, depth+1, handling_front=True)
                else:
                    self.fill_training_example(first_arg, depth+1, handling_front=False)
                self.fill_training_example(second_arg, depth+1, handling_front=False, available_choices=["beh_action"])

            example["args"].append(first_arg)
            example["args"].append(second_arg)

            example["string"] = np.random.choice(["11 and 22",
                                                 "11 and also 22",
                                                 "11 and at the same time 22"])

    def write_out_training_example(self, example):
        written_example = "{}\t{}".format(self.write_out_utterance(example), self.write_out_ast(example))
        return written_example

    def write_out_utterance(self, example):
        try:
            utt_raw = example["string"]
        except:
            return
        subst = []
        for arg in example["args"]:
            subst.append(self.write_out_utterance(arg))

        counter = 1
        while "{}{}".format(str(counter),str(counter)) in utt_raw:
            prefix = utt_raw[:utt_raw.index("{}{}".format(str(counter),str(counter)))]
            suffix = utt_raw[utt_raw.index("{}{}".format(str(counter),str(counter)))+2:]
            utt_raw = prefix + subst[0] + suffix
            del subst[0]
            counter += 1

        return utt_raw

    def write_out_ast(self, example):
        try:
            ex_args = example["args"]
            name = example["name"]
            ast = "( {}".format(name)
            for arg in ex_args:
                arg_str = self.write_out_ast(arg)
                ast += " {}".format(arg_str)
            ast += " )"
            return ast
        except:
            return example

if __name__=='__main__':
    '''

    '''
    generator = Generator()
    generator.load("leaves.json")
    generator.generate(7000)
