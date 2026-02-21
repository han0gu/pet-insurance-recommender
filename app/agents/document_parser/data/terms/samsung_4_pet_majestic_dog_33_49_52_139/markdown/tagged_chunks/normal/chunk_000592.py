from langchain_core.documents import Document

chunk = Document(
    page_content=('- = 200만원\n'
 '- ⑥ 제1항의 「연간」이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의\n'
 '- 기간을 의미합니다.\n'
 '- ⑦ 제2항에도 불구하고 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포\n'
 '- 114 -함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에\n'
 '따라 재가입하는 경우 또는 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외,\n'
 '검사비포함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제5항에 따'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000592',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
