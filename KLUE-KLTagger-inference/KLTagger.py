import os
os.chdir('./20191129_KLTagger/Release')
input_sentence = input("Input Sentence: ")
outfile = open("input.txt",'w')
outfile.write("1\n4\n")
outfile.write(input_sentence)
outfile.write("\n.")
outfile.close()

os.system('trim.bat > nul 2>&1')
"""
infile = open("output.txt",'r')
trivial=1
line = infile.readline()
while line != "":
    if "분석_후보" in line:
        trivial=0
        print("\n")
        print("===============<분석 결과>===============")
        line=infile.readline()


    if trivial==0:
        print(line.rstrip())
    line = infile.readline()
infile.close()
"""

os.system('python trim.py')