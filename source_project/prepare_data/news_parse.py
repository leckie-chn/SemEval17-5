import pandas as pd
import string, os, re
from typing import List


def parse_time(tm_str: str) -> pd.datetime:
    tm_list = [s for s in re.split(" |/|,", tm_str) if s != '']
    month_dict = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12,
    }
    month = month_dict[tm_list[0]]
    day = int(tm_list[1])
    year = int(tm_list[2])
    clock_list = tm_list[3].split(':')
    hour = int(clock_list[0])
    minite = int(clock_list[1])
    if tm_list[4] == 'PM':
        hour += 12
    return pd.datetime(year=year, month=month, day=day, hour=hour, minute=minite)


def parse_news(file_path: str):
    def read_line(f):
        while True:
            line = f.readline()
            if line == '':
                return None
            line = line.strip(string.whitespace)
            if len(line) > 0:
                return line

    tm_str = file_path.split('/')[-2]
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        dash_cnt = 0
        title, url = '', ''
        while True:
            line = read_line(f)
            if line is None:
                break
            if line.startswith('--'):
                dash_cnt += 1
            if dash_cnt == 1:
                line = line.lstrip(string.whitespace + string.punctuation)
                if line != '':
                    title = line
            elif dash_cnt == 4:
                line = line.lstrip(string.whitespace + string.punctuation)
                if line != '':
                    url = line
                    break

    try:
        tm = pd.to_datetime(tm_str.replace('_', '-'))
    except ValueError as e:
        print('Source: {}\nTime String: {}'.format(file_path, tm_str))
        raise (e)
    return title, tm, url


def load_news(news_root):
    tmp = []
    for root, dirs, files in os.walk(news_root):
        source = root.split('/')[-2]
        for name in files:
            fl_name = os.path.join(root, name)
            try:
                title, tm, url = parse_news(fl_name)
            except Exception as e:
                print(e)
                continue
            tmp.append({
                'source': source,
                'time': tm,
                'text': title,
                'url': url,
            })
    return pd.DataFrame.from_records(tmp)


def filter_news(news_df: pd.DataFrame, key_words: List[str]) -> pd.DataFrame:
    company = key_words[0]
    filter_cond = news_df['text'].str.contains(company, case=False)
    for kword in key_words[1:]:
        filter_cond = filter_cond | news_df['text'].str.contains(kword, case=False)
    company_df = news_df.loc[filter_cond]
    company_df = company_df.sort_values(by='date')
    company_df['company'] = company
    return company_df


if __name__ == '__main__':
    df = load_news('/home/zpk/news_data')