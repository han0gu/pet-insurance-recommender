from langchain_core.documents import Document

chunk = Document(
    page_content=('상해를 직접적인 원인으로 치료를 받은 경우에는 보험계약일을 보장개시일(책임개시\n'
 '일)로 합니다. 이 경우 보험계약일은 이 특별약관의 제1회 보험료를 받은 날로 합니다# <예시안내>[ 「반려견 '
 '의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형)」 에 대한 보장개시일\n'
 '(책임개시일) 계산]![image](/image/placeholder)\n'
 '보험계약일 보장개시일(책임개시일)\n'
 '30일주2)\n'
 '2022년 8월 1일 2022년 8월 31일- 주1) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 '
 '합'),
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
 'indexing': {'chunk_id': 'chunk_000458',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
