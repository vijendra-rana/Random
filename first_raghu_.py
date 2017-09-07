
import urllib2
import argparse
import logging
import datetime

def downloadData(str1):
	# this should be supplied at runtime command line argument https://s3.amazonaws.com/cuny-is211-spring2015/birthdays100.csv 
	
    request = urllib2.Request(str1.url)
    response = urllib2.urlopen(request)
    csvData = response.read()
    return csvData


def  processData(str1):
	lines = str1.split("\n")
	my_dict = {}
	logging.basicConfig(filename = 'error.log', level = logging.ERROR,filemode = 'w')
	for idx,line in enumerate(lines[1:]):
		try:
			id,name,date_text = line.split(',')
			date_data = datetime.datetime.strptime(date_text, '%d/%m/%Y')
			my_dict[id] = (name,date_data)
		except ValueError:
			logging.error('Error processing line #%d for ID #%s'%(idx + 1,id))
 	return my_dict

def displayPerson(id, my_dict):
	if id in my_dict:
		print('Person #'+ id + ' is '+ my_dict[id][0] + 'with a birthday of '+ my_dict[id][1].strftime('%Y-%m-%d'))
	else:
		print('No user found with that id')
	    

def main():
	parser = argparse.ArgumentParser(description= 'Download, Process and Lookup stuff')
	parser.add_argument('--url')
	mydata = parser.parse_args()
	result = downloadData(mydata)
	records = processData(result)
	#print(records)
	while True:
		id = raw_input('give an Id to lookup: ')
		if int(id) <= 0:
			exit()
		displayPerson(id,records)

if __name__ == '__main__':
	main()