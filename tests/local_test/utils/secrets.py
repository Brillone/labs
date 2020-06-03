import os
from dotenv import load_dotenv


# load env
load_dotenv('local_test/utils/configs/.env')

# env file
slack_token = os.getenv('SLACK_TOKEN')
