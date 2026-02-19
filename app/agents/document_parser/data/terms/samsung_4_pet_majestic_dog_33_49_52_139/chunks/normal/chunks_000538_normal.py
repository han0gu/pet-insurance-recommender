from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[ 「반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형)」 에 대한 보장개시일 (책임개시일) 계산]\n'
 '주1) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 합 니다. 주2) 암, 백내장, 녹내장, '
 '심장질환, 신장질환, 방광질환 및 각종 결석의 경우 90일'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 99},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive', 'eye', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
