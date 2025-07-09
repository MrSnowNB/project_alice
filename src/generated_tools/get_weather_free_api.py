import requests

def get_weather_free_api(zip_code):
    url = f'https://api.open-meteo.com/v1/forecast?latitude=37.2&longitude=-80.4&current_weather=true'
    response = requests.get(url)
    return response.json()