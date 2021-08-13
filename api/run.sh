#!/bin/bash

export FLASK_APP=api/endpoints.py
if [ "$1" != '-production' ]
then
	export FLASK_ENV=development
else
	export FLASK_ENV=production
fi


flask run --host 0.0.0.0