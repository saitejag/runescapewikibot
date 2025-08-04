from locust import HttpUser, task, between
import json

class ApiTestUser(HttpUser):
    # The wait time between each task will be between 1 and 3 seconds
    wait_time = between(1, 3)
    
    # Define the endpoint to test
    @task
    def post_query(self):
        url = "/api/query"
        
        # Prepare the JSON data to send in the POST request
        payload = {
            "query": "List the quests in Age of Chaos"
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Send POST request
        response = self.client.post(url, data=json.dumps(payload), headers=headers)
        
        # Optional: Check if the response status code is 200 (OK)
        if response.status_code == 200:
            print(f"Request was successful: {response.status_code}")
        else:
            print(f"Request failed: {response.status_code}")


