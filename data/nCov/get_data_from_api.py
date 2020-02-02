import requests
import json
import datetime
import time
import numpy
from collections import defaultdict
import pandas
import configargparse

def get_data_from_api():
    '''
    return a list
    '''
    url = 'https://lab.isaaclin.cn/nCoV//api/area?latest=0'
    session = requests.session()
    r = session.get(url=url)
    content = r.content.decode()
    provinces_date = json.loads(content)
    print('get data from ', url)
    return provinces_date['results']

def get_data_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']

def get_refined_province_date_list(provice_date_list):
    '''
    return a dir with key = province, value = a dir of time series date
    '''
    refined_date = defaultdict(list)
    for province in provice_date_list:
        new_date = {}
        new_date['confirmedCount'] = province['confirmedCount']
        new_date['suspectedCount'] = province['suspectedCount']
        new_date['curedCount'] = province['curedCount']
        new_date['deadCount'] = province['deadCount']
        new_date['updateTime'] = province['updateTime']
        refined_date[province['provinceShortName']].append(new_date)
    for province_name, province_date in refined_date.items():
        refined_date[province_name] = time_as_key(province_date)
    return refined_date

def time_as_key(province_date_list_item):
    '''
    provice date is the value of refined date, which is a list of time series date.
    this function return a dir whose key is updateTime
    '''
    new_dir = {}
    for time_date in province_date_list_item:
        # print(time_date)
        if 'updateTime' in time_date.keys():
            updateTime = time_date.pop('updateTime')
            new_dir[str(updateTime)] = time_date
    sorted_items = sorted(new_dir.items(), key=lambda x: x[0])
    sorted_dir = {}
    for province_name, province_date_dir in sorted_items:
        sorted_dir[province_name] = province_date_dir
    return sorted_dir

def complete(time_list):
    province_date_list = get_data_from_api()
    # province_date_list = get_data_from_file('ncov_2020_01_29.txt')
    refined_data = get_refined_province_date_list(province_date_list)
    completed_date = {}
    for province_name, province_date in refined_data.items():
        completed_date[province_name] = defaultdict(list)
        for setting_time in time_list:
            data_at_timestamp = {'confirmedCount': 0, 'suspectedCount': 0, 'curedCount': 0, 'deadCount': 0}
            for time, time_date in province_date.items():
                # ! the updateTime is about 1000 times larger than time.time()
                # print(setting_time, float(time) / 1000)
                if setting_time >= float(time) / 1000:
                    data_at_timestamp = time_date
                    # print('update')
            for data_kind, data in data_at_timestamp.items():
                completed_date[province_name][data_kind].append(data)
    return completed_date

def get_time_list(start_time, end_time, duration):
    '''
    time:(year, month, day, hour, minute, second)
    '''
    start_year = str(start_time[0]).zfill(4)
    start_month = str(start_time[1]).zfill(2)
    start_day = str(start_time[2]).zfill(2)
    start_hour = str(start_time[3]).zfill(2)
    start_minute = str(start_time[4]).zfill(2)
    start_second = str(start_time[5]).zfill(2)
    start_string = '%s-%s-%s %s:%s:%s' % (start_year, start_month, start_day, start_hour, start_minute, start_second)
    start_timestamp = time.mktime(time.strptime(start_string, '%Y-%m-%d %H:%M:%S'))
    end_year = str(end_time[0]).zfill(4)
    end_month = str(end_time[1]).zfill(2)
    end_day = str(end_time[2]).zfill(2)
    end_hour = str(end_time[3]).zfill(2)
    end_minute = str(end_time[4]).zfill(2)
    end_second = str(end_time[5]).zfill(2)
    end_string = '%s-%s-%s %s:%s:%s' %(end_year, end_month, end_day, end_hour, end_minute, end_second)
    end_timestamp = time.mktime(time.strptime(end_string, '%Y-%m-%d %H:%M:%S'))
    return numpy.arange(start_timestamp, end_timestamp, duration)

def get_df(start_time, end_time, duration, data_kind):
    time_list = get_time_list(start_time, end_time, duration)
    data = complete(time_list)
    given_kind_data = {}
    for province_name, province_data in data.items():
        given_kind_data[province_name] = province_data[data_kind]
    df = pandas.DataFrame(given_kind_data)
    df.index = formated(time_list)
    return df.T

def formated(time_list):
    str_list = []
    for timestamp in time_list:
        str_list.append(time.strftime('%m-%d %H:%M', time.localtime(timestamp)))
    return str_list

def output(duration, index, output_format, data_kind):
    start_time = [2020, 1 ,24, 4, 0, 0]
    now = time.time()
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(now)).split('-')
    end_time = [0, 0, 0, 0, 0, 0]
    for i in range(len(end_time)):
        end_time[i] = int(now[i])
    if duration == 'h':
        duration = 3600
    elif duration == 'd':
        duration = 86400
    df = get_df(start_time, end_time, duration, data_kind + 'Count')
    if output_format == 'csv':
        if index == 'y':
            df.to_csv(data_kind + '.' + output_format)
        elif index == 'n':
            df.to_csv(data_kind + '.' + output_format, header=False, index=False)
    elif output_format == 'xls':
        if index == 'y':
            df.to_excel(data_kind + '.' + output_format)
        elif index == 'n':
            df.to_excel(data_kind + '.' + output_format, header=False, index=False)
    elif output_format == 'no':
        print(df)
    return df
if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add('--duration', type=str, help='h for hour | d for day', default='h')
    p.add('--index', type=str, help='y | n', default='y')
    p.add('--output', type=str, help='format of output: csv|xls|no output', default='no')
    p.add('--kind', type=str, help='data kind : confirmed | dead | cured', default='confirmed')
    opt = vars(p.parse_args())
    output(opt['duration'], opt['index'], opt['output'], opt['kind'])
    # df.to_excel('curedCount_d.xls')

