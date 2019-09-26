#%%
import webbrowser
import urllib
from bs4 import BeautifulSoup
import numpy as np


#%%
validRegion = ['h27v04', 'h27v05', 'h28v05', 'h28v06']
urlH = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD09GA.006/2019.07.'
jpgPath = 'D:\Data\MOD09GQ\jpg/'
urls=[]
for date in range(20, 26):
    for i in range(2):
        urlHead = urlH
        if i == 1:
            urlHead = urlH.replace('T/MOD', 'A/MYD')
        rawurl = urlHead + str('%02d'%date) + '/'
        content = urllib.request.urlopen(rawurl).read().decode('ascii')  #获取页面的HTML
        soup = BeautifulSoup(content, 'lxml')
        url_cand_html=soup.find_all('pre') 
        list_urls=url_cand_html[1].find_all("a") 

        for i in list_urls[1:]:
            url = i.get('href')
            sp = url.split('.')
            region = sp[len(sp)//2 - 1]
            if len(sp) in [6, 8] and region in validRegion:
                print(rawurl + url)
                # if len(sp) == 8:   
                    # name = jpgPath + region + '.' + sp[2] + '.' + sp[1][:3] + '.jpg'
                    # urllib.request.urlretrieve(rawurl + url, name)
                if len(sp) == 6:                
                    urls.append(rawurl + url) #取出链接
                    webbrowser.open(rawurl + url)
np.save('./log/urls.npy', urls)