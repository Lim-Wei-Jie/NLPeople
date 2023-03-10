from selenium import webdriver

#Let's say you are trying to get all announcement pdfs for Jardine Matheson Holdings: https://www.sgx.com/securities/company-announcements?value=JARDINE%20MATHESON%20HLDGS%20LTD&type=company
url="https://www.sgx.com/securities/company-announcements?value=JARDINE%20MATHESON%20HLDGS%20LTD&type=company"
driver = webdriver.Chrome()
driver.get(url)

#pause for a while to let the site load and also to avoid the website from detecting us
import time 
time.sleep(5)

#The main url contains a list of urls - we will collect all the links, then download the pdfs on that page.
elemLs=[]
elems = driver.find_elements_by_xpath("//a[@href]")
for elem in elems:
    x=elem.get_attribute("href")
    #print(x)
    elemLs.append(x)
    
#turns out only urls that contain 'links' has pdfs, so only keeping links that contain pdfs
import re
elemLs = [x for x in elemLs if re.match(r'https://links.sgx.com',x)]

#for each link, download the pdfs.
import urllib.request
for link in elemLs:
    time.sleep(5)
    driver.get(link)
    pdfLs=[]
    elems = driver.find_elements_by_xpath("//a[@href]")
    elems = driver.find_elements_by_xpath("//a[@href]")
    for elem in elems:
        x=elem.get_attribute("href")
        pdfLs.append(x)

    labels = driver.find_elements_by_xpath('/html/body/form/div[3]/div/div[2]/div[3]/dl/dd[2]') #finds time and date
    n=1
    for pdf in pdfLs: 
        urllib.request.urlretrieve(pdf, labels[0].text.replace(':','')+'_'+str(n)+'.pdf')
        n+=1