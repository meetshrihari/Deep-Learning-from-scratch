from argparse import ArgumentParser
import os
import re
from zipfile import ZipFile

def coherency_check(actual_files, desired_files):
    missing_files = []
    ambigous_files = []
    for des_file in desired_files:
        found = 0
        for act_file in actual_files:
            if des_file.lower() == os.path.split(act_file)[1].lower():
                found += 1
        if found == 0:
            missing_files.append(des_file)
        elif found > 1:
            ambigous_files.append(des_file)
    if len(missing_files):
        print("The following files could not be found: ")
        for i, f in enumerate(missing_files):
            print("%d: %20s" %(i, f))
    if len(ambigous_files):
        print("Ambiguities in the file namings were found: ")
        for i, f in enumerate(ambigous_files):
            print("%d: %20s" %(i, f))
    if len(ambigous_files) or len(missing_files):
        return False
    else:
        return True


def dispatch(actual_files, desired_files, output_file):
    files_to_dispatch = []
    for des_file in desired_files:
        for act_file in actual_files:
            if des_file.lower() in act_file.lower():
                files_to_dispatch.append(act_file)
                break
    assert len(files_to_dispatch) == len(desired_files)

    if not output_file.endswith(".zip"):
        output_file += ".zip"

    with ZipFile(output_file,'w') as zip:
        # writing each file one by one
        for file in files_to_dispatch:
            zip.write(file, arcname=os.path.split(file)[1])


def get_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_exercise_number(files):
    unit_test_files = list(filter(lambda x: "numpytests" in x.lower() or "neuralnetworktests" in x.lower() or "pytorchchallengetests", files))
    ids = []
    for file in unit_test_files:
        with open(file, "r") as f:
            lines = f.readlines()
            for l in lines:
                if "ID" in l:
                    ids.append(int(re.search("\d", l).group(0)))
    if len(ids) == 1:
        return ids[0]
    elif len(ids) > 1:
        print("Ambiguities with the exercises. The dispatch looks for the unittest in order to identify exercise at hand. \n"
                        "Please put your stuff in its own folder, together with the corresponding unittest. \n"
                        "Found unittests of the exercises: \n Exercise {}".format('\n Exercise '.join(map(str, ids))))
        exit(1)
    else:
        print("Unittest file could not be found. It might got renamed to something else.")
        exit(1)

if __name__ == "__main__":
    description = "The dispatch tool is a helper script, that conveniently checks all your files for completeness and" \
                  " zips everything together so you can submit just one file. It helps us to have a coherent " \
                  "submissions across all students and also prevents you from forgetting any file to submit." \
                  " If the dispatcher is executed without arguments, it will look for  unittests in the current " \
                  "working directory and will dispatch the corresponding files accordingly into a default zip file " \
                  "submission.zip. We recommend to use the dispatcher explicitly with the -i arguement for the" \
                  " folder that needs to be dispatched and -o arguments for the output zip folder"

    exercise_files = {0: ["generator.py", "main.py", "pattern.py"],
                      1: ["FullyConnected.py", "ReLU.py", "SoftMax.py",
                          "Loss.py", "Optimizers.py", "NeuralNetwork.py"],
                      2: ["FullyConnected.py", "ReLU.py", "SoftMax.py",
                          "Loss.py", "Optimizers.py", "NeuralNetwork.py", "Conv.py", "Flatten.py", "Initializers.py",
                          "Pooling.py"],
                      3: ["FullyConnected.py", "ReLU.py", "Sigmoid.py", "SoftMax.py", "Constraints.py",
                          "Loss.py", "Optimizers.py", "NeuralNetwork.py", "Conv.py", "Flatten.py", "Initializers.py",
                          "Pooling.py", "Base.py", "BatchNormalization.py", "Dropout.py", "LSTM.py", "TanH.py"],
                      4: ["data.py", "train.py", "trainer.py", "model.py"]}

    parser = ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=False,
                        help = "src folder which contains all python files")
    parser.add_argument("-o", "--output", required=False,
                        help="file name of output zip folder")
    args = parser.parse_args()

    if args.input is None and args.output is None:
        print("No arguments were given. Check python dispatch.py --help for further information.\n\n"
              "However shall the content of \n {} \n "
              "be dispatched into \n {} [yes/no]?".format(os.getcwd(), os.path.join(os.getcwd(), "submission.zip")))
        resp = input("")
        if resp.lower() in ["y", "yes"]:
            args.input = os.getcwd()
            args.output = "submission.zip"
        elif resp.lower() in ["n", "no"]:
            exit(0)
        else:
            print("Did not understand your answer. I will exit now!")
            exit(1)

    if not os.path.isdir(args.input):
        raise NotADirectoryError("Path either does not exist or is not a directory")

    files = get_files(args.input)
    files = list(filter(lambda x: not x.endswith(".pyc"), files))
    ex_nr = get_exercise_number(files)
    print("Exercise {} is about to be dispatched".format(ex_nr))
    if not coherency_check(files, exercise_files[ex_nr]):
        print("Dispatching has been stopped. Please check your files.")
    else:
        dispatch(files, exercise_files[ex_nr], args.output)
        print("Your submission contains all it needs and is ready to be submitted")
        print("Please upload {} now to studon".format(args.output))




