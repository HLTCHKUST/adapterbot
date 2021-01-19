import requests
import pprint
from datetime import datetime
from datetime import date 
from collections import defaultdict
from iso3166 import countries
import calendar
import geograpy
import pycountry

def most_frequent(List): 
    return max(set(List), key = List.count) 


def _get_weather(city="",country="",lon=0, lat=0):
      url = "https://community-open-weather-map.p.rapidapi.com/forecast"

      if(lon!= 0 and lat!= 0):
            querystring = {'lat': lat, 'lon': lon, "units":"metric"}
      else:
            querystring = {"q":f"{city},{country}","units":"metric"}

      headers = {
      'x-rapidapi-host': "community-open-weather-map.p.rapidapi.com",
      'x-rapidapi-key': "f65983498fmsh7aeb620012c066ap1a8b06jsn5c9030b8fc6f"
      }

      response = requests.request("GET", url, headers=headers, params=querystring)


      if("400 - Bad Request" not in response.text):
            resp = eval(response.text.replace("false","False").replace("true","True").replace("null","None"))
      else:
            return [], []

      dic_resp = defaultdict(lambda: defaultdict(list))
      for elm in resp['list']:
            date_time_obj = datetime.strptime(elm['dt_txt'], '%Y-%m-%d %H:%M:%S')
            dic_resp[calendar.day_name[date_time_obj.weekday()]]["min"].append(elm['main']['temp_min'])
            dic_resp[calendar.day_name[date_time_obj.weekday()]]["max"].append(elm['main']['temp_max'])
            dic_resp[calendar.day_name[date_time_obj.weekday()]]["wea"].append(elm['weather'][0]['main'])


      today = calendar.day_name[date.today().weekday()]
      meta_weather = [f"today {today}"]
      dict_vix = [f"today {today}"]
      for k, v in dic_resp.items():
            meta_weather.append(f"{resp['city']['name']} {k} {most_frequent(v['wea'])} low {int(min(v['min']))}C high {int(max(v['max']))}C")
            dict_vix.append([f"{resp['city']['name']}",f"{k}",f"{most_frequent(v['wea'])}",f"{int(min(v['min']))}C",f"{int(max(v['max']))}C"])
            # print(f"{resp['city']['name']} {k} {most_frequent(v['wea'])} low {int(min(v['min']))}C high {int(max(v['max']))}C")
      return meta_weather,dict_vix

def get_weather(input_text,lon=0, lat=0):
    places = geograpy.get_place_context(text=input_text)
    if(len(places.cities)==0 and lon!=0):
        meta,dict_vix = _get_weather(city="",country="",lon=lon, lat=lat)
    elif(len(places.cities)>0):
        country_id = pycountry.countries.get(name=list(places.country_cities.keys())[0]).alpha_2
        meta,dict_vix = _get_weather(city=places.cities[0],country=country_id)
    else: 
        meta = []
        dict_vix = []
    err_msg = ""
    if(len(meta)):
        err_msg = "which city are you looking for?"
    print(meta)
    return meta, dict_vix,err_msg
