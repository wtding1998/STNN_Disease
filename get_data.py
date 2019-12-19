import pandas
import codecs  
from django.utils.encoding import smart_text  
import chardet  
import os

def check_file_charset(file):  # get the type of file
    with open(file,'rb') as f:  
        return chardet.detect(f.read())      
  
    return {}  

def convert(folder_path):
    files = os.listdir(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        f_type = check_file_charset(file_path)  
        if f_type and 'encoding' in f_type.keys() and f_type['encoding'] != 'utf-8':  
            try:  
                with codecs.open(file_path, 'rb', f_type['encoding']) as f:  
                    content = smart_text(f.read())  
                with codecs.open(file_path, 'wb', 'utf-8') as f:  
                    f.write(content)  
            except:  
                pass  
    print("convert encoding type successfully!")

def get_province(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, encoding="utf-8") as f:
        data = pandas.read_csv(f)
        data.drop([0,1,2],inplace=True)
    return data["Unnamed: 1"].tolist()

def get_list(folder_path):
    files = os.listdir(folder_path)
    province_set = set()
    for file_name in files:
        new = get_province(file_name, folder_path)
        province_set.update((new))
    return province_set

def province_order():
    return ['北京市', '天津市', '河北省', '山西省', '辽宁省', '黑龙江省', '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', '广    西', '海南省', '重庆市', '四川省', '贵州省', '云南省', '陕西省', '甘肃省', '青海省', '新疆', '内蒙古', '吉林省']

def gather_disease_data(disease_name, folder_path):
    province_list = get_list(folder_path)
    print(province_list)
    index = pandas.Index(data=province_list, name="province")
    data = pandas.DataFrame(index=index)
    files = os.listdir(folder_path)
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path,encoding='utf-8') as f:
            new_data = pandas.read_csv(f)
            new_data.drop([0,1,2], inplace=True)
            new_data = new_data[["Unnamed: 1", "Unnamed: 2"]]
            new_data = new_data.set_index('Unnamed: 1')
            full_month = file_name.replace(disease_name.title()+'_', '')
            full_month = full_month.replace('.xls.csv', '')
            new_data.columns=[full_month]
            new_data.index.name="province"
            data = data.join(new_data)
            print(str(full_month) + " joined to data")
    return data

def output_data_excel(output_path, disease_name, folder_path):
    data = gather_disease_data(disease_name, folder_path)
    # change order of index
    order = province_order()
    province_list = get_list(folder_path)
    rest = [x for x in province_list if x not in order]
    index_order = order + rest
    data = data.reindex(index=index_order)
    # data.fillna(0.1)
    data.to_excel(output_path)
    print("Output data to "+output_path)

if __name__ == "__main__":    
    folder_path = "F:/myPython/python-3/disease_data/influenza/influenza_csv"
    convert(folder_path)
    output_data_excel("F:/myPython/python-3/disease_data/influenza/flu_test.xlsx", "influenza", folder_path)
