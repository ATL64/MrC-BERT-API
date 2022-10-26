## BERT-API

This repo contains code for a small service in docker that provides a REST API for evaluating questions and abstracts with [DistilBERT](https://arxiv.org/abs/1910.01108)
using abstracts from the PubMed database as source of knowledge.

The request needs to be a POST request as follows:

``http://<ip_address_of_vm>:5000/question-pubmed``,

with the following data:

- username: A valid username corresponding to one of the users authorised to use the API.
- password: The username's password.
- question: The question whose answer you're interested in.
- nOutputs: The maximum number of abstracts to be analysed. Currently the algorithm roughly takes 0.2s/abstract.
- min_year: The minimum year the abstract was published in.
- max_year: The maximum year the abstract was published in.

This request enqueues a task on a redis queue and returns the task id in the data of the response object, encoded in the variable `task_id`.
Then, one can check the status of the request with the following GET request:

``http://<ip_address_of_vm>:5000/tasks/<task_id>``,

where `task_id` is the ID returned in the previous POST request.
Once the job is complete, the GET request will return an object with the following data:

- task_id
- task_status
- task_result
  - pmid
  - abstract
  - article_title
  - article_date
  - authors
  - journal
  - answer
  - start_score
  - end_score
  - final_answer
  - total_n_articles
  - keywords

Currently, the IP address of the VM that we're using is 34.107.125.243.


Internally, the program first extracts the keywords from your question and uses PubMed's API to retrieve the relevant abstracts.
Then, it asks the question to all of the abstracts using DistilBERT and returns all the answers and their related scores.


The current VM has the following specs:

- Zone Frankfurt (europe-west3-c)
- 2CPU, 13.75GB Memory
- Ubuntu 18.04
- Disk space 15GB


## Run with docker-compose

The docker image is build using docker-compose, using a combination of the Docker image
redis:6-buster (to manage the redis queues) and the custom Dockerfile present in the repository,
which is based on python:3.7-slim-buster.

In order to run it, you just need to have pip, python3 and docker compose installed, and then execute the command
``docker-compose up -d --build``.

To stop the docker container, just type
``docker-compose stop``,
and to start it again, type
``docker-compose start``.

To view the logs,
``docker-compose logs --tail=30``.
