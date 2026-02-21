from langchain_core.documents import Document

chunk = Document(
    page_content=('검사비포함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제5항에 따\n'
 '라 보험계약이 연장된 경우에는 종전 계약의 보험기간을 연장하는 것으로 보아 제2항\n'
 '을 적용하지 않습니다.⑧ 제3항에도 불구하고 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포\n'
 '함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에\n'
 '따라 재가입하는 경우 또는 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외,'),
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
 'indexing': {'chunk_id': 'chunk_000593',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
