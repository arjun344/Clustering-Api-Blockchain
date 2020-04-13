from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json as simplejson
from django.http import HttpResponse
import os
import datetime
import time
import random
from time import gmtime, strftime
import csv
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
from Source import KMean
import time
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pickle

@api_view(["GET","POST"])
def PredictMiner(request):
	try:
		data = simplejson.loads(request.body)

		address = data['address']
		latest_block_timestamp = int(data['blockdata'][0]['timestamp'])
		ether_balance = float(data['balance'])
		total_ether_block = random.uniform(float(5*len(data['blockdata'])),float(ether_balance)/2.5)

		##calculating total active time
		start_times = data['activestarttime']
		stop_times = data['activestoptime']
		node_config_time = ","+data['nodeconfigtime']

		total_active_time = GetTimeDiff(start_times,stop_times)
		curr = ","+strftime("%Y-%m-%d %H:%M:%S", gmtime())
		total_idle_time = idletime(node_config_time,curr) - total_active_time

		timestamp_diff = int(data['blockdata'][0]['timestamp']) - int(data['blockdata'][len(data['blockdata'])-1]['timestamp'])

		with open('dataset.csv', mode='a') as csv_file:
			fieldnames = ['address','latest_block_timestamp','ether_balance','total_ether_block','total_active_time','total_idle_time','timestamp_diff']
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writerow({'address':address,'latest_block_timestamp':latest_block_timestamp,'ether_balance':ether_balance,'total_ether_block':total_ether_block,'total_active_time':total_active_time,'total_idle_time':total_idle_time,'timestamp_diff':timestamp_diff})


		data = pd.read_csv("dataset.csv") 
		cols = ['total_active_time']
		data[cols] = data[cols].replace({0:np.nan})
		data = data.fillna(data.mean())
		data = data.drop(['address'],axis=1)
		scaler = MinMaxScaler()
		data = scaler.fit_transform(data)
		temp_arr = []
		temp_arr = data.copy()
		X = np.array(temp_arr)
		print(len(data))

		if len(data)%10 == 0:
			optimal_cluster = -1
			ncluster = 1
			for n_cluster in range(2, 10):
				kmeans = KMeans(n_clusters=n_cluster).fit(X)
				label = kmeans.labels_
				sil_coeff = silhouette_score(X, label, metric='euclidean')
				if sil_coeff > optimal_cluster:
					optimal_cluster = sil_coeff
					ncluster = n_cluster
			
			print("For n_clusters={}, The Silhouette Coefficient is {}".format(ncluster, optimal_cluster))
			cllf = KMean.K_Means(k=ncluster, tol=0.001, max_iter=300)
			cllf.fit(X)
			print("saving model....")
			pickle.dump(cllf, open("k_mean_model.sav", 'wb'))
		
		else:
			cllf = pickle.load(open("k_mean_model.sav", 'rb'))

		cluster = cllf.predict(X[-1])
		distance = cllf.getDistance(X[-1])
		target = 0.15

		while distance > target:
			target = target + 0.25
		result = cluster
		time.sleep(5)
		return JsonResponse(str(result),safe=False)

	except ValueError as e:

		return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["GET","POST"])
def GetMiner(request):
	try:
		data = simplejson.loads(request.body)

		if len(data['blockdata']) > 0:
			address = data['address']
			latest_block_timestamp = int(data['blockdata'][0]['timestamp'])
			ether_balance = float(data['balance'])
			total_ether_block = random.uniform(float(5*len(data['blockdata'])),float(ether_balance)/2.5)

			##calculating total active time
			start_times = data['activestarttime']
			stop_times = data['activestoptime']
			node_config_time = ","+data['nodeconfigtime']

			total_active_time = GetTimeDiff(start_times,stop_times)
			curr = ","+strftime("%Y-%m-%d %H:%M:%S", gmtime())
			total_idle_time = idletime(node_config_time,curr) - total_active_time

			timestamp_diff = int(data['blockdata'][0]['timestamp']) - int(data['blockdata'][len(data['blockdata'])-1]['timestamp'])

			with open('dataset.csv', mode='a') as csv_file:
				fieldnames = ['address','latest_block_timestamp','ether_balance','total_ether_block','total_active_time','total_idle_time','timestamp_diff']
				writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
				writer.writerow({'address':address,'latest_block_timestamp':latest_block_timestamp,'ether_balance':ether_balance,'total_ether_block':total_ether_block,'total_active_time':total_active_time,'total_idle_time':total_idle_time,'timestamp_diff':timestamp_diff})

		return JsonResponse(str("written"),safe=False)

	except ValueError as e:

		return Response(e.args[0],status.HTTP_400_BAD_REQUEST)



def GetTimeDiff(start_time , stop_time):
	start_time = start_time.split(',')[1:]
	stop_time = stop_time.split(',')[1:]

	net_diff = 0
	date_format = "%Y-%m-%d"
	for item in range(len(start_time)):
	    start_time[item] = start_time[item].split(' ')[:2]

	for item in range(len(stop_time)):
	    stop_time[item] = stop_time[item].split(' ')[:2]
	    
	for item in range(len(start_time)-1):
	    start_date = start_time[item][0].split('-')
	    start_t = start_time[item][1].split(':')
	    
	    start_year = int(start_date[0])
	    start_month = int(start_date[1])
	    start_day = int(start_date[2])
	    
	    start_hr = int(start_t[0])
	    start_min = int(start_t[1])
	    start_sec = int(float(start_t[2]))
	    
	    
	    stop_date = stop_time[item][0].split('-')
	    stop_t = stop_time[item][1].split(':')
	    
	    stop_year = int(stop_date[0])
	    stop_month = int(stop_date[1])
	    stop_day = int(stop_date[2])
	    
	    stop_hr = int(stop_t[0])
	    stop_min = int(stop_t[1])
	    stop_sec = int(float(stop_t[2]))
	    
	    start = datetime.datetime(start_year, start_month, start_day, start_hr, start_min, start_sec)
	    stop = datetime.datetime(stop_year, stop_month, stop_day, stop_hr, stop_min, stop_sec)
	    
	    diff = stop - start
	    
	    net_diff = net_diff + diff.seconds
	    
	return net_diff


def idletime(start_time , stop_time):
	start_time = start_time.split(',')[1:]
	stop_time = stop_time.split(',')[1:]

	net_diff = 0
	date_format = "%Y-%m-%d"
	for item in range(len(start_time)):
	    start_time[item] = start_time[item].split(' ')[:2]

	for item in range(len(stop_time)):
	    stop_time[item] = stop_time[item].split(' ')[:2]
	    
	for item in range(len(start_time)):
	    start_date = start_time[item][0].split('-')
	    start_t = start_time[item][1].split(':')
	    
	    start_year = int(start_date[0])
	    start_month = int(start_date[1])
	    start_day = int(start_date[2])
	    
	    start_hr = int(start_t[0])
	    start_min = int(start_t[1])
	    start_sec = int(float(start_t[2]))
	    
	    
	    stop_date = stop_time[item][0].split('-')
	    stop_t = stop_time[item][1].split(':')
	    
	    stop_year = int(stop_date[0])
	    stop_month = int(stop_date[1])
	    stop_day = int(stop_date[2])
	    
	    stop_hr = int(stop_t[0])
	    stop_min = int(stop_t[1])
	    stop_sec = int(float(stop_t[2]))
	    
	    start = datetime.datetime(start_year, start_month, start_day, start_hr, start_min, start_sec)
	    stop = datetime.datetime(stop_year, stop_month, stop_day, stop_hr, stop_min, stop_sec)
	    
	    diff = stop - start
	    
	    net_diff = net_diff + diff.seconds
	    
	return net_diff