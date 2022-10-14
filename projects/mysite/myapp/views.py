import base64
import io
from django.http import HttpResponse
from django.shortcuts import render
import socket
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import rc


SERVER_IP = '121.144.179.223'
SERVER_PORT = 1236
SIZE = 1048576
SERVER_ADDR = (SERVER_IP, SERVER_PORT)


def index(request):
    return render(request, 'myapp/index.html')


def result(request):



    sentence = request.POST.get('text')



    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(SERVER_ADDR)  # 서버에 접속
        client_socket.send(sentence.encode())  # 서버에 메시지 전송
        msg = client_socket.recv(SIZE).decode()  # 서버로부터 응답받은 메시지 반환
    # # 받아온 json 결과값으로 결과 처리 (준기코드)

    index_list = []
    word_list = []
    head_list = []

    s = nx.DiGraph()
    pos = {}

    tmp_dict = json.loads(msg)
    print('분석완료')
    plt.figure(figsize=(13, 7))
    # font_location = './NanumGothic.ttf'
    #
    # font_name = fm.FontProperties(fname=font_location).get_name()
    # rc('font', family=font_name)
    # plt.rcParams['font.family'] = 'NanumGothic'
    font_name = fm.FontProperties(fname="C:/Windows/Fonts/malgun.TTF").get_name()
    table = []

    mapping = {}
    for i in range(0, len(tmp_dict['dependency'])):
        t = []
        index_list.append(int(tmp_dict['dependency'][i]['INDEX']))
        word_list.append(tmp_dict['dependency'][i]['WORD_FORM'] + "\n" + tmp_dict['dependency'][i]['DEPREL'])
        head_list.append(int(tmp_dict['dependency'][i]['HEAD']))
        mapping[i] = word_list[i]
        t.append(int(tmp_dict['dependency'][i]['INDEX']))
        t.append(tmp_dict['dependency'][i]['WORD_FORM'])
        t.append(int(tmp_dict['dependency'][i]['HEAD']))
        t.append(tmp_dict['dependency'][i]['DEPREL'])
        table.append(t)
        print(t)
        print(table)

    for i in range(0, len(index_list)):
        # s.add_node(word_list[i])
        s.add_node(i)
        # pos[word_list[i]] = [i, 1]
        # pos.update({word_list[i]: [i, 1]})
        pos.update({i: [i, 1]})
        if head_list[i] != 0:
            if (abs(head_list[i] - index_list[i]) != 0):
                # s.add_edge(word_list[head_list[i] - 1], word_list[i], len=abs(index_list[i] - head_list[i]))
                s.add_edge(head_list[i] - 1, i, len=abs(index_list[i] - head_list[i]))

    # print(mapping)
    # s = nx.relabel_nodes(s, mapping, copy=False)
    # print(s)
    words_limit_pass = True
    font_size = '12'
    node_size = 1000


    if len(index_list) > 15:
        font_size = '7'
        node_size = 500
    elif len(index_list) > 10:
        font_size = '10'
        node_size = 800
    if len(index_list) > 20:
        words_limit_pass = False





    nx.draw_networkx(s, pos, node_shape="s", node_color="white", connectionstyle="arc3, rad=0.7",
                     with_labels=True,font_family=font_name, font_size='0', node_size=node_size)
    nx.draw_networkx_labels(s, pos,mapping, font_family=font_name, font_size=font_size)



    # fig = plt.figure()
    flike = io.BytesIO()
    # fig.savefig(flike)
    plt.savefig(flike, format='png')
    b64 = base64.b64encode(flike.getvalue()).decode('utf-8')
    print('이미지 생성완료')
    # figfile = io.BytesIO()
    # plt.savefig(figfile, format='png')
    # figfile.seek(0)
    # figdata_png = base64.b64encode(figfile.getvalue()).decode()
    # plt.clf()

    #문장입력하고 확인 누르면 실행할 코드 섹션
    context = {'sentence': sentence,
               'msg' : msg,
               'words_limit_pass' : words_limit_pass,
                'graph' : b64,
               'table' : table}
    return render(request, 'myapp/result.html', context)
