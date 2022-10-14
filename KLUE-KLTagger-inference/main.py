import MeCab
import sys
import os
import socket
import json

sys.path.append("./inference.py")

def Eojeol_mecab_mapping(sentence,pos_list):
    f2 = open("./data/dp-v1.1_test.tsv", 'w', encoding='utf-8')
    index2=0
    sentene_Eojeol_list = sentence.split(" ")
    data_list=[]
    b=sentence.split(" ") #문장 어절단위로 잘라서 리스트형태로 반환
    c = [0 for i in range(len(b))] #LEMMA
    c2 = [0 for i in range(len(b))] #POS
    for index in range(len(pos_list)): #본 결과로 c,c2 만들어짐, 형태소분석 결과의 나눠진 개수만큼 for문 돎
        if pos_list[index][1]=='NNBC': #KLUE DP dictionary에 NNBC가 없으므로 NNB로 바꿔줌
            pos_list[index]=list(pos_list[index])
            pos_list[index][1]='NNB' 
            pos_list[index]=tuple(pos_list[index])
        if pos_list[index][1]=='MM': #KLUE DP dictionary에 MM가 없으므로 NND/N/A중 가장 빈도가 많은 MMD로 바꿔줌
            pos_list[index]=list(pos_list[index])
            pos_list[index][1]='MMD' 
            pos_list[index]=tuple(pos_list[index])
        if pos_list[index][1]=='SY': #KLUE DP dictionary에 MM가 없으므로 NND/N/A중 가장 빈도가 많은 MMD로 바꿔줌
            pos_list[index]=list(pos_list[index])
            pos_list[index][1]='SW' 
            pos_list[index]=tuple(pos_list[index])
        if pos_list[index][1]=='SSO' or pos_list[index][1]=='SSC': #KLUE DP dictionary에 MM가 없으므로 NND/N/A중 가장 빈도가 많은 MMD로 바꿔줌
            pos_list[index]=list(pos_list[index])
            pos_list[index][1]='SS'
            pos_list[index]=tuple(pos_list[index])
        if pos_list[index][0] in b[index2]: #태그 결과가 있을 시
            if c[index2]==0:
                c[index2]=pos_list[index][0]
                c2[index2]=pos_list[index][1]
            elif c[index2]!=0:
                c[index2]=c[index2]+" "+pos_list[index][0]
                c2[index2]=c2[index2]+"+"+pos_list[index][1]
        elif (index2+1)<len(b): #4,4
            if pos_list[index][0] in b[index2+1]:
                index2=index2+1
                c[index2]=pos_list[index][0]
                c2[index2]=pos_list[index][1]
        tempc=str(c[index2]).replace(" ","") #다음 pos_list[index]의 단어가 그전의 b[index2]와 겹치는 것을 확인하기 위한 문자열
        if tempc==b[index2]:
            index2=index2+1
    colum_name = "## 칼럼명 : INDEX	WORD_FORM	LEMMA	POS	HEAD	DEPREL"
    f2.write(colum_name)
    f2.write("\n")
    line2="##sentence:\t"+sentence+"\n"
    f2.write(line2)
    for num in range(len(b)):
        data_list.append(num+1)
        data_list.append(sentene_Eojeol_list[num])
        data_list.append(c[num])
        data_list.append(c2[num])
        data_list.append(0)
        data_list.append("NP")
        line="%d\t%s\t%s\t%s\t%d\t%s\n" % (data_list[0],data_list[1],data_list[2],data_list[3],data_list[4],data_list[5])
        f2.write(line)
        if num==(len(b)-1):
            f2.write("\n")
        data_list.clear()

def KLTagger_sentence_parsing(text):
    os.chdir('./20191129_KLTagger/Release')
    outfile = open("input.txt", 'w')
    outfile.write("1\n4\n")
    outfile.write(text)
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
    os.chdir("../../")


if __name__ == "__main__":
    IP = '211.104.246.33'
    PORT = 1236
    SIZE = 1048576
    ADDR = (IP, PORT)

    # 서버 소켓 설정
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(ADDR)  # 주소 바인딩
        server_socket.listen()  # 클라이언트의 요청을 받을 준비
        option=1

        # 무한루프 진입
        try:
            while True:
                client_socket, client_addr = server_socket.accept()  # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환
                sentence = client_socket.recv(SIZE)  # 클라이언트가 보낸 메시지 반환
                print("[{}] message : {}".format(client_addr, sentence.decode()))  # 클라이언트가 보낸 메시지 출력
                #print("평문 입력: ")
                #sentence = input()
                KLTagger_sentence_parsing(sentence.decode())
                with open("inference.py", encoding='utf-8') as f:
                    code = compile(f.read(), "./inference.py", 'exec')
                    exec(code)  # 1차 inference : 지배소 예측
                    print(code)
                with open("inference2.py", encoding='utf-8') as f:
                    code = compile(f.read(), "./inference2.py", 'exec')
                    exec(code)  # 2차 inference : 의존관계 레이블 예측
                    print(code)

                tmp = ""
                """with open("./data/dp-v1.1_test.tsv", "r", encoding='utf-8') as f:
                    lines = f.readlines()
                    for i in lines:
                        tmp+=i"""

                with open("./data/dp-v1.1_test_mecab.json",'rt', encoding='utf-8') as f:
                    json_data=json.load(f)
                    tmp+=json.dumps(json_data,indent=4,ensure_ascii=False)

                client_socket.sendall(tmp.encode())  # 클라이언트에게 응답
                client_socket.close()  # 클라이언트 소켓 종료
        except:
            os.system('python main.py')
    print("mode 입력 : ")
    mode = int(input())

"""   if mode == 1:
        print("모두의 말뭉치")
        with open("./inference3.py", encoding='utf-8') as f:
            code = compile(f.read(), "./inference3.py", 'exec')
            exec(code)  # 모두의 말뭉치 inference : 의존관계 레이블 예측

   elif mode == 2:
        print("평문 입력: ")
        sentence = input()
        eojeol_split_list = sentence.split(" ")
        KLTagger_sentence_parsing(sentence)
        with open("inference.py",encoding='utf-8') as f:
            code = compile(f.read(), "./inference.py", 'exec')
            exec(code) # 1차 inference : 지배소 예측
            print(code)
        with open("inference2.py", encoding='utf-8') as f:
            code = compile(f.read(), "./inference2.py", 'exec')
            exec(code)  # 2차 inference : 의존관계 레이블 예측
            print(code)"""

#덕분에 우리 과학의 미래가 밝습니다.

