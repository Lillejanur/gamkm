#!/usr/bin/env python

colorbrewer = {}

#-------------Qualitative---------------

#12 colors

#Paired with first four is colorblind safe.
Paired = {'light_blue': '#a6cee3',
          'blue': '#1f78b4',
          'light_green': '#b2df8a',
          'green': '#33a02c',
          'pink': '#fb9a99',
          'red': '#e31a1c',
          'light_orange': '#fdbf6f',
          'orange': '#ff7f00',
          'light_purple': '#cab2d6',
          'purple': '#6a3d9a',
          'yellow': '#ffff99',
          'brown': '#b15928'}

#9 colors

#Nice red, blue, green colors
Set1 = {'red': '#e41a1c',
        'blue': '#377eb8',
        'green': '#4daf4a',
        'purple': '#984ea3',
        'orange': '#ff7f00',
        'yellow': '#ffff33',
        'brown': '#a65628',
        'pink': '#f781bf',
        'grey': '#999999'}

#8 colors

#Dark2 with 3 colors is colorblind safe
Dark2 = {'turquoise': '#1b9e77',
         'rust': '#d95f02',
         'lavender': '#7570b3',
         'magenta': '#e7298a',
         'lime_green': '#66a61e',
         'yellow': '#e6ab02',
         'brown': '#e6ab02',
         'grey': '#e6ab02'}

#Set2 with 4 colors is colorblind safe.
#Brighter version of Dark2
Set2 = {'turquoise': '#66c2a5',
        'rust': '#fc8d62',
        'lavender': '#8da0cb',
        'magenta': '#e78ac3',
        'lime_green': '#a6d854',
        'yellow': '#ffd92f',
        'brown': '#e5c494',
        'grey': '#b3b3b3'}

#Not very nice
Accent = {'green': '#7fc97f',
          'lavender': '#beaed4',
          'peach': '#fdc086',
          'yellow': '#ffff99',
          'blue': '#ffff99',
          'magenta': '#f0027f',
          'brown': '#f0027f',
          'grey': '#f0027f'}

#--------------Diverging------------------

RdYlBu_5 = {'r2': '#d7191c',
            'r1': '#fdae61',
            '0': '#ffffbf',
            'b1': '#abd9e9',
            'b2': '#2c7bb6'}

RdYlBu_6 = {'r3': '#d73027',
            'r2': '#fc8d59',
            'r1': '#fee090',
            'b1': '#e0f3f8',
            'b2': '#91bfdb',
            'b3': '#4575b4'}

RdYlBu_7 = RdYlBu_6.copy()
RdYlBu_7['0'] = '#ffffbf'

RdYlBu_8 = {'r4': '#d73027',
            'r3': '#f46d43',
            'r2': '#fdae61',
            'r1': '#fee090',
            'b1': '#e0f3f8',
            'b2': '#abd9e9',
            'b3': '#74add1',
            'b4': '#4575b4'}

RdYlBu_9 = RdYlBu_8.copy()
RdYlBu_9['0'] = '#ffffbf'

RdYlBu_10 = RdYlBu_8.copy()
RdYlBu_10['r5'] = '#a50026'
RdYlBu_10['b5'] = '#313695'

RdYlBu_11 = RdYlBu_10.copy()
RdYlBu_11['0'] = '#ffffbf'

#-----------------Sequential-------------------
Blues = {}
Blues[3] = ['#deebf7','#9ecae1','#3182bd']
Blues[4] = ['#eff3ff','#bdd7e7','#6baed6','#2171b5']
Blues[5] = ['#eff3ff','#bdd7e7','#6baed6','#3182bd','#08519c']
Blues[6] = ['#eff3ff','#c6dbef','#9ecae1','#6baed6','#3182bd','#08519c']
Blues[7] = ['#eff3ff','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594']
Blues[8] = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#084594']
Blues[9] = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']

Greens = {}
Greens[3] = ['#e5f5e0','#a1d99b','#31a354']
Greens[4] = ['#edf8e9','#bae4b3','#74c476','#238b45']
Greens[5] = ['#edf8e9','#bae4b3','#74c476','#31a354','#006d2c']
Greens[6] = ['#edf8e9','#c7e9c0','#a1d99b','#74c476','#31a354','#006d2c']
Greens[7] = ['#edf8e9','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#005a32']
Greens[8] = ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#005a32']
Greens[9] = ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b']

Greys = {}
Greys[3] = ['#f0f0f0','#bdbdbd','#636363']
Greys[4] = ['#f7f7f7','#cccccc','#969696','#525252']
Greys[5] = ['#f7f7f7','#cccccc','#969696','#636363','#252525']
Greys[6] = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#636363','#252525']
Greys[7] = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
Greys[8] = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
Greys[9] = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000']

Oranges = {}
Oranges[3] = ['#fee6ce','#fdae6b','#e6550d']
Oranges[4] = ['#feedde','#fdbe85','#fd8d3c','#d94701']
Oranges[5] = ['#feedde','#fdbe85','#fd8d3c','#e6550d','#a63603']
Oranges[6] = ['#feedde','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603']
Oranges[7] = ['#feedde','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#8c2d04']
Oranges[8] = ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#8c2d04']
Oranges[9] = ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704']

Purples = {}
Purples[3] = ['#efedf5','#bcbddc','#756bb1']
Purples[4] = ['#f2f0f7','#cbc9e2','#9e9ac8','#6a51a3']
Purples[5] = ['#f2f0f7','#cbc9e2','#9e9ac8','#756bb1','#54278f']
Purples[6] = ['#f2f0f7','#dadaeb','#bcbddc','#9e9ac8','#756bb1','#54278f']
Purples[7] = ['#f2f0f7','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#4a1486']
Purples[8] = ['#fcfbfd','#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#4a1486']
Purples[9] = ['#fcfbfd','#efedf5','#dadaeb','#bcbddc','#9e9ac8','#807dba','#6a51a3','#54278f','#3f007d']

Reds = {}
Reds[1] = ['#cb181d'] #Mine
Reds[2] = ['#fcae91','#cb181d'] #Mine
Reds[3] = ['#fee0d2','#fc9272','#de2d26']
Reds[4] = ['#fee5d9','#fcae91','#fb6a4a','#cb181d']
Reds[5] = ['#fee5d9','#fcae91','#fb6a4a','#de2d26','#a50f15']
Reds[6] = ['#fee5d9','#fcbba1','#fc9272','#fb6a4a','#de2d26','#a50f15']
Reds[7] = ['#fee5d9','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']
Reds[8] = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#99000d']
Reds[9] = ['#fff5f0','#fee0d2','#fcbba1','#fc9272','#fb6a4a','#ef3b2c','#cb181d','#a50f15','#67000d']
