# WARNING: Running this script may consume your openai account's
# credit. You may only have limited free credit in your account.
#
# Please install openai client library first:
# pip install openai
#
import openai
from openai import OpenAI
import os

# Returns your OpenAI API key.
#
# If you do not have one, go to chatgpt api key pages and create one,
# and then create a file chatgpt_key.txt in your home dir and
# put the contents into that file.
def get_api_key():
    # Get the home directory
    home_dir = os.path.expanduser('~')

    # Define the path to the file in the home directory
    file_path = os.path.join(home_dir, 'chatgpt_key.txt')

    # Read the file
    with open(file_path, 'r') as file:
        file_content = file.read()

    return file_content

def chat_with_gpt(prompt):
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt 
                }
            ]
        )

        return completion.choices[0].message
    except Exception as e:
        return str(e)


# Example usage
if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = get_api_key()
    prompt = "Tell me who is Adar in Rings of Power in 30 words."
    print("ChatGPT response:", chat_with_gpt(prompt))
