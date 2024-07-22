import numpy as np
import json
import string
from nbformat import read, write, NO_CONVERT
import os

def extract_mcq_from_raw_text(raw_question,
                              opts="abcd",
                              code_itentifier="\n\npython\nCopy code\n",
                              opt_formatter=lambda x: f"\n{x}) "):
    options = {}
    for s,s_next in zip(opts[:-1],opts[1:]):
        search_start = opt_formatter(s)
        search_end = opt_formatter(s_next)
        assert search_start in raw_question, f"search_start={search_start} not in raw_question"
        assert search_end in raw_question, f"search_end={search_end} not in raw_question"
        start = raw_question.index(search_start)+len(search_start)
        end = raw_question.index(search_end)
        options[s] = raw_question[start:end]
    search_start = f"\n{opts[-1]}) "
    assert search_start in raw_question, f"search_start={search_start} not in raw_question"
    start = raw_question.index(search_start)+len(search_start)
    options[opts[-1]] = raw_question[start:]
    pre_options_str = raw_question[:raw_question.index(opt_formatter(opts[0]))]
    if code_itentifier in pre_options_str:
        code_start = pre_options_str.index(code_itentifier)+len(code_itentifier)
        code = pre_options_str[code_start:]
        question = pre_options_str[:code_start-len(code_itentifier)]
    else:
        code = None
        question = pre_options_str
    return question,options,code

def format_markdown_mcq(question,options,
                        code=None,
                        qnum=None,
                        options_are_code=False,
                        shuffle_options=True,
                        gt="a",
                        opts="abcd",
                        with_newline=True):
    assert all([opt in opts for opt in options.keys()]), f"options={options.keys()} not in opts={opts}"
    assert gt in options.keys(), f"gt {gt} not in options={options.keys()}"

    has_num = (qnum is not None)
    qnum_formatted = f"{str(qnum)}.    " if has_num else ""
    mcq_string = qnum_formatted+question
    nl = "\n" if with_newline else "<br/>"
    if code is not None:
        mcq_string += f"{nl}```python{nl}{code}{nl}```"
    else:
        mcq_string += nl

    for opt in opts:
        mcq_string += f"{nl}  *   {opt}) [OPT{opt}]"
    if shuffle_options:
        perm = np.random.permutation(len(opts))
        old_to_new = {k: v for k,v in zip(opts,[opts[p] for p in perm])}
    else:
        old_to_new = {k: k for k in opts}

    new_opts = {old_to_new[k]: v for k,v in options.items()}
    new_gt = old_to_new[gt]
    extra = "`" if options_are_code else ""
    for opt in opts:
        mcq_string = mcq_string.replace(f"[OPT{opt}]",extra+new_opts[opt]+extra)
    if qnum is not None:
        gt = f"{qnum}: '{new_gt}'," if has_num else str(new_gt)

    return mcq_string,gt

def format_all_questions(quiz_file,
                         add_status=True,
                         save_txt=None,
                         verbose=True,
                         raise_errors=False,
                         with_newline=True):
    with open(quiz_file) as f:
        quiz = json.load(f)
    gts = {}
    full_formatted = ""
    for i in range(len(quiz)):
        try:
            if len(quiz[i].get("chatgpt_text",""))>0:
                    raw_question = quiz[i]["chatgpt_text"]
                    while raw_question.endswith("\n"):
                        raw_question = raw_question[:-1]
                    question,options,code = extract_mcq_from_raw_text(raw_question)
            elif len(quiz[i].get("question",""))>0:
                assert "options" in quiz[i].keys(), f"options missing for question with id={quiz[i]['id']}"
                assert isinstance(quiz[i]["options"],dict), f"options should be dict for question with id={quiz[i]['id']}"
                question = quiz[i]["question"]
                options = quiz[i]["options"]
                code = quiz[i].get("code",None)
            else:
                quiz[i]["status"] = "missing"
                continue
        except:
            if verbose:
                print(f"Failed to extract question with id={quiz[i]['id']}")
            quiz[i]["status"] = "error_extract"
            if raise_errors:
                raise
            else:
                continue
        nl = "\n" if with_newline else "<br/>"
        try:
            formatted,gt = format_markdown_mcq(question,
                                               options,
                                               code=code,
                                               qnum=i+1,
                                               options_are_code=quiz[i].get("options_are_code",False),
                                               gt=quiz[i].get("gt","a"),
                                               with_newline=with_newline)
            gts[i] = gt
            full_formatted += formatted+nl+nl
            quiz[i]["status"] = "success"
        except:
            if verbose:
                print(f"Failed to format question with id={quiz[i]['id']}")
            quiz[i]["status"] = "error_format"
            if raise_errors:
                raise
            else:
                continue
        
    if len(full_formatted)>0:#remove last newlines
        full_formatted = full_formatted[:-len(nl+nl)]
    if add_status:
        with open(quiz_file,"w") as f:
            json.dump(quiz,f,indent=4)
    if save_txt is not None:
        #delete if it exists
        if os.path.exists(save_txt):
            os.remove(save_txt)
        with open(save_txt,"w") as f:
            f.write(full_formatted)
    if verbose:
        status = {quiz[i]["status"] for i in range(len(quiz))}
        status_counts = {s: sum([1 for i in range(len(quiz)) if quiz[i]["status"]==s]) for s in status}
        print(f"{len(quiz)} total questions")
        for k,v in status_counts.items():
            print(f"{v} questions with status={k}")
    return full_formatted,gts

exercise_type_to_description = {"1": "Write complete function",
                                "1a": "Mostly from text description",
                                "1b": "Mostly from unit tests",
                                "1c": "Mostly from visual guidance",
                                "1d": "Mostly from math",

                                "2": "Modify existing code",
                                "2a": "Debug/Fix code",
                                "2b": "Fill missing line(s)",
                                "2c": "Fill code from skeleton",
                                "2d": "Add functionality",
                                "2e": "Simplify/Remove redundancy",

                                "3": "Translate code to/from",
                                "3a": "Math",
                                "4b": "Natural language",
                                "3c": "Pseudocode",

                                "4": "Understand code (no coding)",
                                "4a": "What happens when code is run",
                                "4b": "Which is correct/proofread",
                                "4c": "Explain error/exception",
                                "4d": "Answer coding related questions"}

def assert_info_dict_format(info_dict):
    """
    asserts the info dict is correctly formatted. E.g. has all the required keys
    with the correct types
    
    info_dict = {"name": "Nice sign",
             "num_exercise": "4.1",
             "author": "jloch",
             "source": "modified_cp23",
             "difficulty": 2,
             "type": "1b",
             "sub_exercises": ["a", "b", "c", "d"],
             "is_advanced": False}"""
    if isinstance(info_dict, str):
        #use json.loads on everything between { and }
        assert info_dict.count("{")==1, f"Found more than one {'{'} in cell. expected only one when given a string."
        assert info_dict.count("}")==1, f"Found more than one {'}'} in cell. expected only one when given a string."
        decode_str = info_dict[info_dict.index("{"):info_dict.index("}")+1]
        decode_str = decode_str.replace("False","false").replace("True","true")
        try:
            info_dict = json.loads(decode_str)
        except Exception as e:
            print(f"Could not decode json from the string: {info_dict}.")
            raise ValueError(f"Could not decode json in info_dict string. Error: {e}")
    req_keys = ["name","num_exercise","author","source","difficulty","type"]
    assert all([k in info_dict.keys() for k in req_keys]), f"info_dict missing keys: "+str([k for k in req_keys if k not in info_dict.keys()])
    assert type(info_dict["name"]) == str, f"info_dict['name'] should be str"
    assert type(info_dict["num_exercise"]) == str, f"info_dict['num_exercise'] should be str"
    assert all([item.isdigit() for item in info_dict["num_exercise"].split(".")]), f"info_dict['num_exercise'] should be of form 'a.b' where a and b are numbers"
    valid_sources = ["modified_cp23","new_original_content","prev_year"]
    assert info_dict["source"] in valid_sources, f"info_dict['source'] should be one of {valid_sources}. found {info_dict['source']}"
    assert type(info_dict["difficulty"]) == int, f"info_dict['difficulty'] should be int"
    assert 1<=info_dict["difficulty"]<=10, f"info_dict['difficulty'] should be between 1 and 10"
    assert type(info_dict["type"]) == str, f"info_dict['type'] should be str"
    if info_dict["type"].startswith("1"):
        hopefully_letters = info_dict["type"][1:]
        valid_letters = [v[1:] for v in exercise_type_to_description.keys() if v.startswith("1")]
        valid_letters = "".join(valid_letters)
        assert all([l in valid_letters for l in hopefully_letters]), f"If info_dict['type'] is '1[something]' then something should be one or more of {valid_letters}. found {hopefully_letters}"
    else:
        assert info_dict["type"] in exercise_type_to_description.keys(), f"info_dict['type'] should be one of {exercise_type_to_description.keys()}"
    assert type(info_dict["is_advanced"]) == bool, f"info_dict['is_advanced'] should be bool"
    assert type(info_dict["is_coding_drill"]) == bool, f"info_dict['is_coding_drill'] should be bool"

def verify_docstring(doc):
    """
    asserts the docstring is correctly formatted. E.g. has all the required keys and correct types.
    Example of correct formatting a function, common_prefix(word1, word2):
    doc = {"Description": "Return the longest string so that both word1, and word2 begin with that string",
        "Parameters": {"word1": {"Type": "str", "Description": "The first word"},
                       "word2": {"Type": "str", "Description": "The second word"}},
        "Returns": {"Type": "str", "Description": "The longest common prefix."}}
    """
    allowed_keys = ["Description","Parameters","Returns"]
    allowed_keys_parameters = ["Type","Description","Default"]
    allowed_keys_returns = ["Type","Description"]
    assert all([k in doc.keys() for k in allowed_keys]), f"doc missing keys: "+str([k for k in allowed_keys if k not in doc.keys()])
    assert type(doc["Description"]) == str, f"doc['Description'] should be str"
    assert type(doc["Parameters"]) == dict, f"doc['Parameters'] should be dict"
    for param_name,v in doc["Parameters"].items():
        assert isinstance(v, dict), f"Values in doc['Parameters'] should be dicts of str keys and str values. Found: {type(v)} for doc['Parameters'][{param_name}]"
        assert all([k in allowed_keys_parameters for k in v.keys()]), f"doc['Parameters'][{param_name}] found wrong key: "+str([k for k in v.keys() if k not in allowed_keys_parameters])+f". only allowed keys are: {allowed_keys_parameters}"
        assert all([type(v[k]) == str for k in v.keys()]), f"Values in doc['Parameters'][{param_name}] should be dicts of str keys and str values"
    assert type(doc["Returns"]) == dict, f"doc['Returns'] should be dict"
    assert all([k in allowed_keys_returns for k in doc["Returns"].keys()]), f"doc['Returns'] found wrong key: "+str([k for k in doc["Returns"].keys() if k not in allowed_keys_returns])+f". only allowed keys are: {allowed_keys_returns}"
    assert all([type(v[k]) == str for k in doc["Returns"].keys()]), f"Values in doc['Returns'] should be dicts of str keys and str values"

def ascending_reorder_of_exercises(ipynb_path,
                                   weeknum="auto",
                                   replace_all_exercise_names=True,
                                   exercise_prefix = lambda weeknum: f"## Exercise {weeknum}.",
                                   exercise_suffix = ":",
                                   verbose=True,
                                   do_save=True):
    """
    Processes a notebook and reorders the exercises in ascending order. A new
    exercise is assumed to start with a markdown cell that starts with "## Exercise [weeknum].[exercisenum]:".
   
    Parameters:
    ipynb_path (str): path to the notebook file.
    weeknum (str): week number of the notebook. If "auto" it will 
        try to extract it from the filename.
    replace_all_exercise_names (bool): if True, searches through all
        cells to replace the exercise number in children cells.
    exercise_prefix (str): prefix of the exercise markdown cell. e.g. 
        if the exercise starts with "## 4.1:" then the prefix is "## ".
    exercise_suffix (str): suffix of the exercise markdown cell. e.g. 
        if the exercise starts with "## 4.1:" then the suffix is ":".
    verbose (bool): if True, prints the number of exercises found and the
        number of exercises reordered.
    do_save (bool): if True, saves the notebook after reordering the exercises.
    """
    if weeknum=="auto":
        hopefully_week_num = ipynb_path.split("/")[-1].split("_")[0]
        if hopefully_week_num.isdigit():
            weeknum = int(hopefully_week_num)
            if verbose:
                print(f"Found weeknum={weeknum} in filename: {ipynb_path}")
        else:
            raise ValueError(f"weeknum='auto' but could not find a number in the filename: {ipynb_path}. Try setting weeknum manually or making sure the notebook name starts with an integer followed by an underscore.")
    # Read the notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = read(f, as_version=NO_CONVERT)
    
    # Initialize variables
    
    exercise_prefix = exercise_prefix(weeknum)
    valid_exercise_strings = [str(i) for i in range(100)]
    exercise_counter = 0

    # Loop through the notebook cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            lines = cell['source'].split('\n')
            if lines and lines[0].startswith(exercise_prefix):
                if verbose and replace_all_exercise_names and exercise_counter>0:
                    print(f"Replaced {replacements} instances of exercise {s_old} with {s_new}.")
                replacements = 0
                exercise_counter += 1
                start = len(exercise_prefix)
                stop = lines[0].find(exercise_suffix)
                if stop < 0:
                    raise ValueError(f"Invalid format for exercise in codeblock starting with line: {lines[0]}, expected suffix={exercise_suffix}.")
                exercise_str = lines[0][start:stop]
                if exercise_str not in valid_exercise_strings:
                    raise ValueError(f"Invalid format for exercise: {lines[0]}, expected an integer between 0 and 99 after '{exercise_prefix}' and before '{exercise_suffix}'.")
                s_old = f"{weeknum}.{exercise_str}"
                if exercise_str != str(exercise_counter):
                    #first replace just the title exercise number
                    new_exercise_str = str(exercise_counter)
                    lines[0] = lines[0][:start]+new_exercise_str+lines[0][stop:]
                    cell['source'] = '\n'.join(lines)
                    s_new = f"{weeknum}.{new_exercise_str}"
                else:
                    s_new = s_old
        if replace_all_exercise_names and exercise_counter>0:
            #replace all instances of the exercise str in both markdown and code cells
            replacements += cell['source'].count(s_old)
            cell['source'] = cell['source'].replace(s_old,s_new)
    if verbose and replace_all_exercise_names and exercise_counter>0:
        print(f"Replaced {replacements} instances of exercise {s_old} with {s_new}.")
    if verbose:
        print(f"Found {exercise_counter} exercises and reordered them.")
    if do_save:
        # Write the updated notebook
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            write(notebook, f)

def format_x_subexercise(ipynb_path, exercise_str="4.14",prefix="###"):
    # Read the notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = read(f, as_version=NO_CONVERT)

    # Initialize variables
    exercise_prefix = exercise_str
    alphabet = string.ascii_lowercase
    subexercise_counter = 0
    max_subexercises = len(alphabet)

    # Loop through the notebook cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            lines = cell['source'].split('\n')
            if lines and lines[0].startswith(prefix):
                first_word = lines[0].split()[1] if len(lines[0].split()) > 1 else ""
                if first_word.startswith(exercise_prefix):
                    suffix = first_word[len(exercise_prefix):]
                    if not suffix.isalpha():
                        raise ValueError(f"Invalid format for subexercise: {first_word}, expected {exercise_prefix} followed by one or more letters.")

                    # Check if subexercise_counter exceeds the alphabet length
                    if subexercise_counter >= max_subexercises:
                        raise ValueError("Exceeded the limit of subexercises (past 'z').")

                    # Replace with new letter
                    new_suffix = alphabet[subexercise_counter]
                    new_first_word = f"{exercise_prefix}{new_suffix}"
                    lines[0] = lines[0].replace(first_word, new_first_word, 1)
                    cell['source'] = '\n'.join(lines)
                    subexercise_counter += 1

    # Write the updated notebook
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        write(notebook, f)
    return f"Updated subexercise labels up to {exercise_prefix}{alphabet[subexercise_counter-1]}"

def change_weeknum(old_weeknum,
                    new_weeknum,
                    rename_inside_exercise=True,
                    rename_files=True,
                    rename_folder=True,
                    exercise_prefix = lambda weeknum: f"## Exercise {weeknum}.",
                    exercise_suffix = ":",
                    replace_all_exercise_names=True):
    """
    Function to change the week number of a folder of exercises/qiuz files.
    It can also rename the files inside the exercises notebook.

    Parameters:
    old_weeknum (str,int): the old week number. Assumes files are 
        contained in a folder named e.g. "W04".
    new_weeknum (str,int): the new week number.
    rename_inside_file (bool): if True, renames the exercise numbers
        inside the file.
    """
    if old_weeknum==new_weeknum:
        return
    foldername = f"W{int(old_weeknum):02d}"
    valid_start_ends = [(f"{int(old_weeknum):02d}_","FULL.ipynb"),
                        (f"{int(old_weeknum):02d}_","quiz.json"),
                        (f"{int(old_weeknum):02d}_","quiz.txt")]
    md_startswith_old = exercise_prefix(old_weeknum)
    md_startswith_new = exercise_prefix(new_weeknum)
    exercise_str = None
    replacements = 0
    for filename in os.listdir(foldername):
        basename = os.path.basename(filename)
        for start,end in valid_start_ends:
            if basename.startswith(start) and basename.endswith(end):
                if end.endswith(".ipynb") and rename_inside_exercise:
                    ipynb_path = os.path.join(foldername,basename)
                    with open(ipynb_path, 'r', encoding='utf-8') as f:
                        notebook = read(f, as_version=NO_CONVERT)
                    is_first_cell = True
                    for cell in notebook['cells']:
                        if cell['cell_type'] == 'markdown':
                            if is_first_cell:
                                is_first_cell = False
                                cell['source'] = cell['source'].replace(f"W{old_weeknum}:",f"W{new_weeknum}:")
                            if cell['source'].startswith(md_startswith_old):
                                lines = cell['source'].split('\n')
                                if lines and lines[0].startswith(md_startswith_old):
                                    lines[0] = lines[0].replace(md_startswith_old,md_startswith_new)
                                    cell['source'] = '\n'.join(lines)
                                exercise_str = lines[0][len(md_startswith_old):lines[0].find(exercise_suffix)]
                                assert exercise_str.isdigit(), f"Invalid format for exercise: {lines[0]}, expected an integer after '{md_startswith_old}' and before '{exercise_suffix}'."
                        if replace_all_exercise_names and (exercise_str is not None):
                            #replace all instances of the exercise str in both markdown and code cells
                            replacements += cell['source'].count(f"{old_weeknum}.{int(exercise_str)}")
                            cell['source'] = cell['source'].replace(f"{old_weeknum}.{int(exercise_str)}",
                                                                    f"{new_weeknum}.{int(exercise_str)}")
                    
                        with open(ipynb_path, 'w', encoding='utf-8') as f:
                            write(notebook, f)
                if rename_files:
                    new_basename = basename.replace(start,f"{int(new_weeknum):02d}_")
                    os.rename(os.path.join(foldername,basename),os.path.join(foldername,new_basename))
                
    if rename_folder:
        os.rename(foldername,f"W{int(new_weeknum):02d}")


def check_info_exists(ipynb_path,
                    weeknum="auto",
                    exercise_prefix = lambda weeknum: f"## Exercise {weeknum}.",
                    exercise_suffix = ":",
                    verbose=True,
                    error_on_no_suffix=False,
                    error_on_bad_number_format=False,
                    info_startswith_str="##INFO"):
    if weeknum=="auto":
        hopefully_week_num = ipynb_path.split("/")[-1].split("_")[0]
        if hopefully_week_num.isdigit():
            weeknum = hopefully_week_num
            if verbose:
                print(f"Found weeknum={weeknum} in filename: {ipynb_path}")
        else:
            raise ValueError(f"weeknum='auto' but could not find a number in the filename: {ipynb_path}. Try setting weeknum manually or making sure the notebook name starts with an integer followed by an underscore.")

    # Read the notebook
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        notebook = read(f, as_version=NO_CONVERT)
    inside_exercise = False
    info_found = False
    exercise_prefix = exercise_prefix(weeknum)
    print(exercise_prefix)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            lines = cell['source'].split('\n')
            if lines and lines[0].startswith(exercise_prefix):
                start = len(exercise_prefix)
                stop = lines[0].find(exercise_suffix)
                if stop < 0:
                    if error_on_no_suffix:
                        raise ValueError(f"Invalid format for exercise in codeblock starting with line: {lines[0]}, expected suffix={exercise_suffix}.")
                    else:
                        continue
                exercise_str = lines[0][start:stop]
                if not exercise_str.isdigit():
                    if error_on_bad_number_format:
                        raise ValueError(f"Invalid format for exercise: {lines[0]}, expected an integer after '{exercise_prefix}' and before '{exercise_suffix}'.")
                    else:
                        continue
                #we have found an exercise
                if inside_exercise:
                    if not info_found:
                        raise ValueError(f"Found exercise {exercise_str} but no info cell.")
                if verbose:
                    print(f"Found exercise {exercise_str}.")
                inside_exercise = True
                info_found = False
        elif inside_exercise and cell['cell_type'] == "code":
            if cell['source'].startswith(info_startswith_str):
                info_found = True
                if verbose:
                    print(f"Found info cell for exercise {exercise_str}.")
                assert_info_dict_format(cell['source'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Format subexercise labels in a notebook.')
    parser.add_argument("--process", type=int, help="Process to run.",default=0)
    parser.add_argument('--ipynb_path', type=str, help='Path to the notebook file.', 
                        default="./04_iteration_and_string_FULL.ipynb")
    parser.add_argument('--exercise_str', type=str, help='Exercise string to format (e.g. "4.14").', 
                        default="4.14")
    parser.add_argument('--prefix', type=str, help='Prefix for exercise headers.', 
                        default="###")
    args = parser.parse_args()
    
    if args.process==0:
        print("Process 0: Format subexercise labels in a notebook.")
        print(format_x_subexercise(args.ipynb_path, args.exercise_str, args.prefix))
    elif args.process==1:
        print("Process 1: Reorder exercises in a notebook.")
        print(ascending_reorder_of_exercises(args.ipynb_path))
    elif args.process==2:
        print("Process 2: Change week number in a folder of exercises.")
        old_weeknum = input("Enter the old week number: ")
        new_weeknum = input("Enter the new week number: ")
        change_weeknum(old_weeknum, new_weeknum)
    else:
        raise ValueError(f"Invalid process: {args.process}.")
    