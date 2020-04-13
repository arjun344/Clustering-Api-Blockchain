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
from datetime import datetime
import time
import random

@api_view(["GET","POST"])
def GetMiner(request):
	try:
		data = simplejson.loads(request.body)
		if os.path.exists("flag"):
			chk_flag = open("flag","r")
			chk_flag = chk_flag.read()
		else:
			chk_flag = open("flag","w")
			chk_flag.write("1")	
			chk_flag = open("flag","r")	
			chk_flag = chk_flag.read()

		if int(chk_flag) == 1:

			filename = "Miners_to_be_processed"

			if os.path.exists(filename):
				append_write = 'a' 
			else:
				append_write = 'w'
				
			temp_data = open(filename,append_write)
			temp_data.write("," + data['address'])
			temp_data.close()

			if os.path.exists("time"):
				current_time = time.time()
			else:
				time_data = open("time","w")
				current_time = time.time()
				time_data.write(str(current_time))

		else:
			
			filename = "Miners_to_be_discarded"

			if os.path.exists(filename):
				append_write = 'a' 
			else:
				append_write = 'w' 

			temp_data = open(filename,append_write)
			temp_data.write("," + data['address'])
			temp_data.close()


		start_time = open("time", "r")
		curr_time = time.time()
		start_time = start_time.read()
		print(float(start_time.strip()))

		elapsed = curr_time - float(start_time)

		while elapsed < 20:

			start_time = open("time", "r")
			start_time = start_time.read()
			start_time = start_time.strip()
			elapsed = time.time() - float(start_time)

			if elapsed > 10:
				chk_flag = open("flag","w")
				chk_flag.write("0")


		miners = open("Miners_to_be_processed", "r")
		miners = miners.read()
		miners = miners.split(',')
		miners = miners[1:]

		result = miners[random.randint(0,len(miners))]

		if os.path.exists("Miners_to_be_processed"):
			os.remove("Miners_to_be_processed")
		
		if os.path.exists("Miners_to_be_discarded"):
			os.remove("Miners_to_be_discarded")

		if os.path.exists("time"):
			os.remove("time")

		if os.path.exists("flag"):
			os.remove("flag")

		chk_flag = open("flag","w")
		chk_flag.write("1")


		return JsonResponse(result,safe=False)

	except ValueError as e:

		return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

