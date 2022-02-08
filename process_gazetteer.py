import pandas as pd
import spacy
import en_core_web_sm
import ja_core_news_trf
import difflib

def get_device_info(ocr_result, nlp):
    ocr_doc = nlp(ocr_result)
    pre_name = ''
    device_name = ''
    device_no = ''
    for token in ocr_doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
        if token.pos_ == 'NUM':
            if len(device_name) == 0:
                #device_name += token.text
                pre_name += token.text
            else:
                device_no += token.text
        elif token.pos_ in ['NOUN', 'PROPN', 'ADV'] and token.tag_ != '空白':
            if token.text == '次' or token.text == '災':
            #if token.text == '次':
                pre_name += '次'
                continue
            if len(device_no) == 0:
                device_name += token.text
            else:
                device_no += token.text
    
    return pre_name, device_name, device_no

def get_top_similarities(device_name, gz_sents, nlp):
    device_doc = nlp(device_name)
    top5_sims = [[0, ''], [0, ''], [0, ''], [0, ''], [0, '']]
    for gz_sent in gz_sents:
        gz_doc = nlp(gz_sent)
        sim = device_doc.similarity(gz_doc)
        for rank, top_sim in enumerate(top5_sims):
            if sim > top_sim[0]:
                next_rank = len(top5_sims) - 1
                while (next_rank != rank):
                    top5_sims[next_rank] = top5_sims[next_rank - 1]
                    next_rank -= 1

                top5_sims[rank] = [sim, gz_sent]
                break
    return top5_sims

def main():
    ocr_results_path = './data/tesseract_results_wrong.txt'
    ocr_results = []
    with open(ocr_results_path, 'r') as f:
        ocr_results = f.readlines()

    file_path = './data/Monju.xlsx'
    jp_df = pd.read_excel(file_path, sheet_name='1.日本語順190710')
    gz_sents = jp_df['日本語（文字コード順）']

    #print(jp_df.columns)

    nlp = spacy.load('ja_ginza')
    
    for ocr_result in ocr_results:
        print('OCR result: {}'.format(ocr_result))
        pre_name, device_name, device_no = get_device_info(ocr_result, nlp)
        if len(pre_name) == 1:
            pre_name += '次'

        top5_sims = get_top_similarities(device_name, gz_sents, nlp)
        print(top5_sims)
        device_name = top5_sims[0][1]
        #device_name = get_sim_device_name(device_name, gz_sents)
        print('Pre name: {}'.format(pre_name))
        print('Device name: {}'.format(device_name))
        print('Device No.: {}'.format(device_no))

def get_sim_device_name(device_name, gz_sents):
    results = []
    start = 0
    for i in range(len(device_name)):
        s = device_name[start:i+1]
        print(s)
        top_sim = 0
        top_gz = ''
        for gz in gz_sents:
            sim = difflib.SequenceMatcher(None, s, gz).quick_ratio()
            if sim > top_sim:
                top_sim = sim
                top_gz = gz
        #print(top_sim)
        if top_sim > 0.8 or (top_sim > 0.7 and len(s) == len(top_gz)):
            print('Top candidate: {}, {}'.format(top_sim, top_gz))

            not_top = False
            if i < len(device_name) - 1:
                for gz in gz_sents:
                    sim_next = difflib.SequenceMatcher(None, device_name[start:i+2], gz).quick_ratio()
                    if sim_next > top_sim:
                        not_top = True
                        break

                    if sim_next > 0.7 and len(device_name[start:i+2]) == len(gz):
                        not_top = True
                        break

            if not not_top:
                results.append(top_gz)
                start = i+1
                print('Top: {}, {}'.format(top_sim, top_gz))

    return ''.join(results)

def test():
    #a = 'ナトリゥウムオーバフロー系'
    a = '格納容書調エイ率試験装置'
    a = 'ナトリゥウムオーバフロー系'
    a = 'メンテナンス冷元系'
    #a = '炎ナトリウム純化系ドレンタンク'
    a1 = 'ナトリゥウム'
    b = 'ナトリウム'
    b = '格納容器'
    c = 'オーバフロー系'
    c = '漏エイ率試験'

    results = []
    start = 0
    gzs = ['格納容器', '漏エイ率試験', 'オーバフロー系', 'ナトリウム', 'メンテナンス冷却系', 'ナトリウム純化系', 'ドレンタンク', '純化系']
    for i in range(len(a)):
        s = a[start:i+1]
        print(s)
        top_sim = 0
        top_gz = ''
        for gz in gzs:
            sim = difflib.SequenceMatcher(None, s, gz).quick_ratio()
            if sim > top_sim:
                top_sim = sim
                top_gz = gz
        print(top_sim)
        if top_sim > 0.7:
            not_top = False
            for gz in gzs:
                sim_next = difflib.SequenceMatcher(None, a[start:i+2], gz).quick_ratio()
                if sim_next > top_sim:
                    not_top = True
                    break

                if sim_next > 0.7 and len(a[start:i+2]) == len(gz):
                    not_top = True
                    break

            if not not_top:
                results.append(top_gz)
                start = i

    print(''.join(results))

if __name__ == '__main__':
    main()
    #test()