# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:40:32 2022

@author: Majid Majeed Qreshi
"""


# Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("GDP_Growth_World_Development_Data.csv")

# Remove None Value
df = df.dropna(axis=0) 
df

# Rename Column of Dataset
df.rename(columns={'Country Name': 'country'}, inplace=True)

# Divide dataset for Population growth & Energy Consumption
gdp  = df.loc[df['Series Code'] == "NY.GDP.MKTP.KD.ZG"]
adj  = df.loc[df['Series Code'] == "NY.ADJ.DRES.GN.ZS"]

# Remove unnecessary column
gdp.drop(['Country Code','Series Name', 'Series Code'], axis=1, inplace=True)
adj.drop(['Country Code','Series Name', 'Series Code'], axis=1, inplace=True)

gdp.set_index('country', inplace=True)
adj.set_index('country', inplace=True)

'''
Graph Plot Functions
    - These function get different parameters
    - Dataframe 
    - Countries Name 
    - Years 
    - Title of graph
    - filename that saved in local file
'''

def graph_plot(df, countries, years, title, filename):
    # figure size is the graph image size
    # This is a simple graph.
    df.loc[countries, years].T.plot(figsize=(14, 8)) 
    title = plt.title(title) 
    plt.savefig(filename)
    plt.show()

def graph_plot2(df, countries, years, title,  filename):
    # figure size is the graph image size
    # This graph type is Area.
    df.loc[countries, years].T.plot(kind='area', figsize=(14, 8))
    title = plt.title(title)
    plt.savefig(filename)
    plt.show()

def graph_bar(df, country, years, title, filename):
    # figure size is the graph image size
    # This is Bar graph
    df.loc[country, years].plot(kind='bar',figsize=(14, 8))
    title = plt.title(title)
    plt.savefig(filename)
    plt.show()

def graph_barh(df, countries, years, title, filename):
    # figure size is the graph image size
    # This is Barh graph
    df.loc[countries, years].transpose().plot(kind='barh', figsize=(20, 14), stacked=False)
    title = plt.title(title)
    plt.savefig(filename)
    plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
# Create a heatmap from a numpy array and two lists of labels.
#     - data - A 2D numpy array of shape (M, N).
#     - row_labels - A list or array of length M with the labels for the rows.
#     - col_labels - A list or array of length N with the labels for the columns.

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def graph_heatmap(df, countries, years, title, filename):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    im, cbar = heatmap(df[years].T, df[years], countries, ax=ax,
                       cmap="viridis", cbarlabel=title)
    plt.savefig(filename)
    fig.tight_layout()
    plt.show()



# Years & Country Information
# - Year used 1980-2020
# - Countries used 'United Kingdom', 'Australia', 'South Africa', 'New Zealand'



# years = list(population_growth.columns)
years = ['1980 [YR1980]', '1985 [YR1985]', '1990 [YR1990]', '1995 [YR1995]', '2000 [YR2000]', 
         '2005 [YR2005]', '2010 [YR2010]', '2015 [YR2015]']

countries = ['United Kingdom', 'Australia', 'South Africa', 'New Zealand']



# GDP Growth Line Graph 
graph_plot(gdp, countries, years, "GDP growth (annual %)", "gdp_1.jpg")

# GDP Growth Bar Graph
graph_bar(gdp, countries, years, 'GDP growth (annual %)', "gdp_2.jpg")

# GDP Growth HeatMap
graph_heatmap(gdp, countries, years, "GDP growth (annual %)", "gdp_3.jpg")

# Natural Resources Depletion Line Graph
graph_plot(adj, countries, years, 'Natural Resources Depletion (% of GNI)', "adj_1.jpg")

# Energy Consumption Bar Graph
graph_bar(adj, countries, years, 'Natural Resources Depletion (% of GNI)', "adj_2.jpg")

# Energy Consumption Heat Graph
graph_heatmap(adj, countries, years, "Natural Resources Depletion (% of GNI)", "adj_3.jpg")


gdp[years].T


adj[years].T