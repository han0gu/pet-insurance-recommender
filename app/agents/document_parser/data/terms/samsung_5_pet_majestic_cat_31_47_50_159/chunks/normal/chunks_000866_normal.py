from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 신체부위\n'
 '"신체부위" 라 함은 ① 눈 ② 귀 ③ 코 ④ 씹어먹거나 말하는 기능 ⑤ 외모 ⑥ 척추 (등뼈) ⑦ 체간골 ⑧ 팔 ⑨ 다리 ⑩ 손가락 ⑪ '
 '발가락 ⑫ 흉 · 복부 장기 및 비뇨생식기 ⑬ 신경계 · 정신행동의 13개 부위를 말하며, 이를 각각 동일한 신체부위라 한다. 다 만, '
 '좌 우의 눈, 귀, 팔, 다리, 손가락, 발가락은 각각 다른 신체부위로 본다.\n'
 '3. 기타'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['eye',
                             'dental',
                             'skin',
                             'joint',
                             'urinary',
                             'digestive',
                             'other']},
 'indexing': {'chunk_id': 'chunk_000866',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
