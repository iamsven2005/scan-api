# locustfile.py
from locust import HttpUser, task, between

class FileUploadUser(HttpUser):
    wait_time = between(1, 5)  # wait between 1 to 5 seconds between tasks

    @task
    def upload_file(self):
        with open("sample.txt", "rb") as f:
            files = {"file": ("sample.txt", f, "text/plain")}
            self.client.post("/scan", files=files)
