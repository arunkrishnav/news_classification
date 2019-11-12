import json
import logging

import requests

from authentication.models import Auth
from authentication.constants import API_BASE_URL, USERNAME, TEAM_TOKEN, CONTENT_TYPE

AUTH_LOG = logging.getLogger('authentication')


class APIHandler(object):
    """
    Handles API requests to main server
    """

    @staticmethod
    def login():
        """
        Performs logging-in to the main server and creates an authentication entry
        :return: authentication object
        """
        data = {
            "name": USERNAME,
            "token": TEAM_TOKEN
        }
        header = {
            "Content-Type": CONTENT_TYPE
        }
        try:
            auth = Auth.objects.get(name=USERNAME, team_token=TEAM_TOKEN)
        except Auth.DoesNotExist:
            auth = Auth()
        response = requests.post(API_BASE_URL+"login/", headers=header, data=json.dumps(data)).json()
        auth_token = response.get('auth_token')
        team_details = response.get('team')
        team_id = team_details["id"]

        if auth_token and team_details and team_id:
            auth.name = USERNAME
            auth.team_token = TEAM_TOKEN
            auth.team_id = team_id
            auth.auth_token = auth_token
            auth.save()
            return auth
        else:
            AUTH_LOG.error('Failed to get authentication details.')

    @staticmethod
    def get_auth():
        """
        :return: authentication object
        """
        auth_set = Auth.objects.all()
        if auth_set:
            auth = auth_set.last()
        else:
            auth = APIHandler.login()
        return auth

    @staticmethod
    def get_current_iteration():
        """
        Calls the current iteration API and validates the response.
         Returns current-iteration if response is valid.
        :return: current_iteration or None
        """
        try:
            auth = APIHandler.get_auth()
            header = {
                "Content-Type": CONTENT_TYPE,
                "TOKEN": auth.auth_token
            }
            response = requests.get(API_BASE_URL + "current-iteration/", headers=header)
            if response.status_code == 200:
                return response.json().get('current-iteration')
            elif response.status_code == 401:
                log = 'Got `unauthorized` response from server for `current-iteration/` API.\n' \
                      'Deleting all existing tokens.'
                print(log)
                AUTH_LOG.info(log)
                Auth.objects.all().delete()
        except Exception as error:
            AUTH_LOG.error("Failed to get iteration details for current iteration: %s", error)

    @staticmethod
    def get_problem_set():
        """
        Calls the problem-set API and validates the response.
         Returns problem-set if response is valid.
        :return: problem_set or None
        """
        try:
            auth = APIHandler.get_auth()
            header = {
                "Content-Type": CONTENT_TYPE,
                "TOKEN": auth.auth_token
            }
            response = requests.get(API_BASE_URL+"problem-set/", headers=header)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                log = 'Got `unauthorized` response from server for `problem-set/` API.\n' \
                      'Deleting all existing tokens.'
                print(log)
                AUTH_LOG.info(log)
                Auth.objects.all().delete()
        except Exception as error:
            AUTH_LOG.error("Failed to fetch problem-set: %s", error)

    @staticmethod
    def get_problem(problem_id):
        """
        Calls the problem details API and validates the response.
         Returns problem details if response is valid.
        :param problem_id: id of the problem
        :return: problem_details or None
        """
        try:
            auth = APIHandler.get_auth()
            header = {
                "Content-Type": CONTENT_TYPE,
                "TOKEN": auth.auth_token
            }
            response = requests.get(API_BASE_URL + "problem/{}/".format(problem_id), headers=header)
            if response.status_code == 200:
                return response.json().get("problem")
            elif response.status_code == 401:
                log = 'Got `unauthorized` response from server for `problem/{}/` API.\n' \
                      'Deleting all existing tokens.'.format(problem_id)
                print(log)
                AUTH_LOG.info(log)
                Auth.objects.all().delete()
        except Exception as error:
            AUTH_LOG.error("Failed to fetch problem-details: %s", error)

    @staticmethod
    def submit_result(data):
        """
        Calls the problem details API and validates the response.
         Returns problem details if response is valid.
        :param data: data to be submitted
        :return: success_case [bool]
        """
        problem_id = data.get('problem')
        response_success = False
        try:
            auth = APIHandler.get_auth()
            header = {
                "Content-Type": CONTENT_TYPE,
                "TOKEN": auth.auth_token
            }
            data['team'] = auth.team_id
            response = requests.post(API_BASE_URL + "problem/submit/", headers=header, data=json.dumps(data))
            if response.status_code == 200:
                response_success = True
            elif response.status_code == 401:
                log = 'Got `unauthorized` response from server for `problem/submit/` API for problem: {}.\n' \
                      'Deleting all existing tokens.'.format(problem_id)
                print(log)
                AUTH_LOG.info(log)
                Auth.objects.all().delete()
            return response_success, response
        except Exception as error:
            AUTH_LOG.error("Failed to fetch problem-details: %s", error)
