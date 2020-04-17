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
import math
from collections import defaultdict
import networkx as nx

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

def load_pickle(url):
    response = requests.get(url)
    gzip_file = io.BytesIO(response.content)
    with gzip.open(gzip_file, 'rb') as f:
        return pickle.load(f)

references = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/references_20200413.pkl.gz")
df = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/df_part_20200413.pkl.gz")
sentences = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/sentences_body.pkl.gz")
sorted_sim = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/sorted_sim_body.pkl.gz")
reduced_vec = load_pickle("https://storage.googleapis.com/mlstudy-phys/20200415_patent_search/body_reduced_vec.pkl.gz")

sentence_id = {v:k for v, k in enumerate(sentences.keys())}

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
    html.H1("Patent Map"),
    html.P(["This is a demonstration of feature extraction using ALBERT (", html.A("arXiv:1909.11942", href="https://arxiv.org/abs/1909.11942", target="_blank"), 
    ")."]),
    html.P(["In this demo, patent literature relating to faster R-CNN (", html.A("arXiv:1506.01497", href="https://arxiv.org/abs/1506.01497", target="_blank"), 
    ") are presented."]),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Citation Network', value='tab-1'),
        dcc.Tab(label='Embedding Map', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1
    elif tab == 'tab-2':
        return tab2

tab1 = html.Div([
     html.Div([
         html.P("Links between patent literature represent forward citation. Size and color of marker represent citation count."),
         dcc.Graph(id='graph')
        ],style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
         html.P("Enter the number of references you want the link to appear in. Links between the input sentence and patent literature represent connections with similar patent literature."),
         dcc.Input(
                id="top_N",
                type="number",
                value=10,
                placeholder="input number",
                ),
         html.P("Select the sentence you want to search."),
         html.P("In this demo, underlying technologies behind the Faster R-CNN are listed as the input sentences."),
         dcc.Dropdown(
                id="target_sentence",
                options=[{'label': t, 'value': i} for i, t in enumerate(sentences.keys())],
                value=0
            ),
         html.Blockquote(id='text_output', style={'backgroundColor':"#DCDCDC"}),
         html.H2("Selected references"),
         html.P(["Shift+click will accumulate the selected reference. You can also use ", html.I("Box Select "), "or ", html.I("Lasso Select "), "to select multiple references."]),
         html.Div([html.Div(id='table')])
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
])

tab2 = html.Div([
     html.Div([
         html.P("Size and color of marker represent citation count."),
         dcc.Graph(id='map')
        ],style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
         html.P("Select the sentence you want to show."),
         html.P("In this demo, underlying technologies behind the Faster R-CNN are listed as the input sentences."),
         dcc.Dropdown(
                id="target_sentence_2",
                options=[{'label': t, 'value': i} for i, t in enumerate(sentences.keys())],
                value=[0],
                multi=True
            ),
         html.Blockquote(id='text_output_2', style={'backgroundColor':"#DCDCDC"}),
         html.H2("Selected references"),
         html.P(["Shift+click will accumulate the selected reference. You can also use ", html.I("Box Select "), "or ", html.I("Lasso Select "), "to select multiple references."]),
         html.Div([html.Div(id='table_2')])
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
])

@app.callback(
    Output(component_id='graph', component_property='figure'),
    [Input(component_id='top_N', component_property='value'),
    Input(component_id='target_sentence', component_property='value')])
def update_figure(top_N, value):
    edge_list = copy.copy(tmp_list)
    sorted_result = copy.copy(sorted_sim[sentence_id[value]])

    j = 0
    for m in range(50):
        n = node_dict[sorted_result[m][0]]
        if n > 0:
            edge_list.append( (target, n) )
            j += 1
        if j >= top_N:
            break

    G = nx.Graph()
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, k=1.2/math.log2(len(edge_list)), pos={target:(0,0)}, fixed=[target] )

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
    Output(component_id='text_output', component_property='children'),
    [Input(component_id='target_sentence', component_property='value')]
)
def update_output_div(value):
    return html.P(sentences[sentence_id[value]])

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

@app.callback(
    Output(component_id='map', component_property='figure'),
    [Input(component_id='target_sentence_2', component_property='value')])
def update_map(value):
    index = list(range(433)) + [elm + 433 for elm in value]
    vec_df = reduced_vec.iloc[index]
    node_trace = go.Scatter(
        x=vec_df["PCA_1"], 
        y=vec_df["PCA_2"], 
        mode='markers',
        hoverinfo='text',
        text = ["{}, {}".format(num, a) for num, a in zip(vec_df["id"], vec_df["assignee"])],
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Portland',
            reversescale=True,
            color=vec_df["forward_citation_count"],
            size=vec_df["size"],
            colorbar=dict(
                thickness=15,
                title='Forward Citation Count',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                clickmode='event+select',
                margin=dict(b=20,l=5,r=5,t=40),
                # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                # width=1000, 
                height=800
                ))

    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                x=vec_df["PCA_1"][l],
                y=vec_df["PCA_2"][l],
                xref="x",
                yref="y",
                text=sentence_id[l-433],
                showarrow=True,
                arrowhead=7,
                ax=10*math.cos(math.atan2(vec_df["PCA_2"][l], vec_df["PCA_1"][l])) 
                * max([(vec_df["PCA_1"][l]**2 + vec_df["PCA_2"][l]**2)**0.5, 0.1])**-1,
                ay=-10*math.sin(math.atan2(vec_df["PCA_2"][l], vec_df["PCA_1"][l])) 
                * max([(vec_df["PCA_1"][l]**2 + vec_df["PCA_2"][l]**2)**0.5, 0.1])**-1
            )
        for l in [elm + 433 for elm in value]]
    )
    return fig

@app.callback(
    [Output(component_id='table_2', component_property='children')],
    [Input('map', 'selectedData')])
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
