from langchain_core.documents import Document

chunk = Document(
    page_content=('.\n'
 '<예시안내>\n'
 '[「반려견 의료비(치과및구강질환포함)(수술당일)(재가입형)」에 대한 보장개시일(책임개시일) 계 산]\n'
 '주1) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 합 니다. 주2) 암, 백내장, 녹내장, '
 '심장질환, 신장질환, 방광질환 및 각종 결석의 경우 90일\n'
 '<유의사항>\n'
 '[수술]\n'
 "동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상 해 또는 질병 치료를 위하여 수의사법 "
 '제 17조(개설)에서 규정한 국내의 동물병원에서 수의사의'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['dental', 'digestive', 'eye', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000686',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
