# -*- coding: utf-8 -*-
import io
import os
import sys
import re
import string
import requests
import pickle
import gzip
import copy 
import multiprocessing
import math
from collections import defaultdict
import networkx as nx

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

def load_pickle(url):
    response = requests.get(url)
    gzip_file = io.BytesIO(response.content)
    with gzip.open(gzip_file, 'rb') as f:
        return pickle.load(f)

references = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/references_20200413.pkl.gz")
df = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/df_part_20200413.pkl.gz")
if not os.path.exists("./config.json"):
    response = requests.get("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/config.json")
    with open("./config.json", 'wb') as saveFile:
        saveFile.write(response.content)
if not os.path.exists("./pytorch_model.bin"):
    response = requests.get("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/pytorch_model.bin")
    with open("./pytorch_model.bin", 'wb') as saveFile:
        saveFile.write(response.content)

model = AlbertForSequenceClassification.from_pretrained('./')
# print(model)

def abstract_preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    text = " ".join([w for w in text.split() if not re.match(r"^[0-9]{1,5}[a-z]$|^[0-9]{1,5}.*[0-9]$|^\(.*\)$|\\n|\\t|^\\", w)])    
    return re.sub("^abstract ","", text)

def set_pair(input_sentence):
    input_sentence = abstract_preprocess(input_sentence)
    pair = []
    for num in df.index:
        text_a = df.at[num, "preprocessed_abstract"]
        text_b = input_sentence
        pair.append([text_a, text_b])
    return pair
    
class PairDataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        one_token = self.input_ids[idx]
        one_token_type = self.token_type_ids[idx]
        one_mask = self.attention_mask[idx]

#         device = torch.device("cuda")
        device = torch.device("cpu")
        sample = {'input_ids': torch.tensor(one_token, device=device), 
                'token_type_ids': torch.tensor(one_token_type, device=device), 
                'attention_mask': torch.tensor(one_mask, device=device)
                }
        return sample

BATCH = 4
def calc_rank(target_sentence):
    torch.set_num_threads(multiprocessing.cpu_count())
    model.eval()
    prob = []

    test_pair = set_pair(target_sentence)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    token = tokenizer.batch_encode_plus(test_pair, add_special_tokens=True, 
    return_token_type_ids=True, max_length=512, return_attention_masks=True, pad_to_max_length=True)
    test_dataset = PairDataset(token["input_ids"], token["token_type_ids"], token["attention_mask"])
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            input_ids = data["input_ids"]
            token_type_ids = data["token_type_ids"]
            attention_mask = data["attention_mask"]

            logits, = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            prob += torch.softmax(logits, dim=1).tolist()

            if i%10==9:
                print("{} done".format((i+1)*BATCH))
    result = [(num, elm[1]) for num, elm in zip(df.index, prob)]
    sorted_result = sorted(result, key=lambda x:x[1], reverse=True)
    return sorted_result

node_dict=defaultdict(int)
tmp_list = []
i = 1

for num in df.index:
    if not node_dict[num]:
        node_dict[num] = i
        dic = references[num]
        i += 1

    num_list = [url.split(sep="/")[-2] for url in dic["ForwardReferences"]]
    for n in num_list:
        if not node_dict[n]:
            if n in df.index:
                node_dict[n] = i
                i += 1
                tmp_list.append( (node_dict[num], node_dict[n]) )

i += 1
target = i

num_to_id_dict = {v:k for k,v in node_dict.items()}

app.layout = html.Div([
    html.H1("Patent citation network"),
    html.P(["This is a demonstration of using ALBERT (", html.A("arXiv:1909.11942", href="https://arxiv.org/abs/1909.11942", target="_blank"), 
    ") to extract literature similar to the input sentences and display it on the network graph."]),
    html.P(["In this demo, patent literature relating to faster R-CNN (", html.A("arXiv:1506.01497", href="https://arxiv.org/abs/1506.01497", target="_blank"), 
    ") are used."]),
     html.Div([
         dcc.Graph(id='graph')
        ],style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
         html.P("Enter the number of references you want the link to appear in."),
         dcc.Input(
                id="top_N",
                type="number",
                value=10,
                placeholder="input number",
                ),
         html.P("Input the sentence you want to search."),
         dcc.Textarea(
             id='target_sentence',
             value='',
             style={'width': '90%', 'height': 200},
             ),
         html.P(["Due to machine resource limitations, the process takes ", html.B("10 minitues"), " or more."]),
         html.Button('Submit', id='button', n_clicks=0),
         html.H2("Selected references"),
         html.P(["Shift+click will accumulate the selected reference. You can also use ", html.I("Box Select "), "or ", html.I("Lasso Select "), "to select multiple references."]),
         html.Div([html.Div(id='table')])
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
])

@app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='top_N', component_property='value'),
    Input(component_id='button', component_property='n_clicks')],
    [State('target_sentence', 'value')])
def update_figure(top_N, n_clicks, target_sentence):
    edge_list = copy.copy(tmp_list)
    if n_clicks > 0:
        if target_sentence != "":
            sorted_result = calc_rank(target_sentence)

            j = 0
            for elm in sorted_result:
                n = node_dict[elm[0]]
                if n > 0:
                    edge_list.append( (target, n) )
                    j += 1
                if j >= top_N:
                    break

    G = nx.Graph()
    G.add_edges_from(edge_list)
    if target_sentence != "":
        pos = nx.spring_layout(G, k=1.2/math.log2(len(edge_list)), pos={target:(0,0)}, fixed=[target] )
    else:
        pos = nx.spring_layout(G, k=1.2/math.log2(len(edge_list))) 

    for node in G.nodes:
        G.nodes[node]['pos'] = pos[node]

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Portland',
            reversescale=True,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Forward Citation Count',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    forward_citation_count = []
    node_text = []
    for node in G.nodes():
        if node == target:
            forward_citation_count.append(10)
            node_text.append("Input sentence")
            continue
        forward_citation_count.append(len(references[num_to_id_dict[node]]["ForwardReferences"] ))
        text = "{}".format(num_to_id_dict[node]) +", " + "{}".format(df[df.index==num_to_id_dict[node]]["assignee"].values[0]) 
        node_text.append(text)


    node_trace.marker.color = forward_citation_count
    node_trace.marker.size = [min(5 + (elm)**0.6, 30) for elm in forward_citation_count]
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=True,
                hovermode='closest',
                clickmode='event+select',
                margin=dict(b=20,l=5,r=5,t=40),
                # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                # width=1000, 
                height=800
                )
                )

    annotations = []
    if target_sentence != "":
        X, Y = G.nodes[target]['pos']
        theta = math.atan2(Y, X)
        r = max([(X**2 + Y**2)**0.5, 0.1])
        d = dict(
            x=X,
            y=Y,
            text="Input sentence",
            showarrow=True,
            arrowhead=7,
            ax=10*math.cos(theta) * r**-1,
            ay=-10*math.sin(theta) * r**-1)
        annotations.append(d)

    fig.update_layout(
        showlegend=False,
        annotations=annotations)

    return fig

@app.callback(
    [Output(component_id='table', component_property='children')],
    [Input('graph', 'selectedData')])
def display_click_data(selectedData):
    if selectedData is not None:
        num_list = [elm["text"].split(sep=",")[0] for elm in selectedData["points"] if elm["text"].split(sep=",")[0] != "Input sentence"]
        url_list = ["https://patents.google.com//patent/" + num + "/en" for num in num_list]

        tmp = [html.Tr([html.Th(elm) for elm in ["URL", "assignee", "priority_date", "abstract"]])]
        for num, url in zip(num_list, url_list):
            elm_list = [url] + df[df.index == num][["assignee", "priority_date", "abstract"]].values[0].tolist()
            tmp.append(html.Tr([html.Td(html.A(elm.split(sep="/")[-2], href=elm, target="_blank")) if i == 0 else html.Td(elm) for i, elm in enumerate(elm_list)]))
        out_table = html.Table(tmp)
        return [out_table]
    else:
        return [html.Tr([html.Td("") for _ in range(4)])]

if __name__ == '__main__':
    app.run_server(debug=False)
