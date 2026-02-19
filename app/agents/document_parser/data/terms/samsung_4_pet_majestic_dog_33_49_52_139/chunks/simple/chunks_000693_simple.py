from langchain_core.documents import Document

chunk = Document(
    page_content=('⑧ 제3항에도 불구하고 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포 함)(재가입형) 특별약관 제27조 (특별약관의 '
 '재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, '
 '검사비포함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제5항에 따 라 보험계약이 연장된 경우에는 '
 '보장개시일(책임개시일)은 이 특별약관의 보험계약일 로 봅니다.\n'
 '제 2조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 115},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000693',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
