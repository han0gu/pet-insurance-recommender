from langchain_core.documents import Document

chunk = Document(
    page_content=('배변 · 배뇨 | · 배설을 돕기 위해 설치한 의료장치나 외과적 시술물을 사용함에 있어 타인의 계 속적인 도움이 필요한 상태, 또는 '
 '지속적인 유치도뇨관 삽입상태, 방광루, 요도 루, 장루상태(20%) · 화장실에 가서 변기위에 앉는 일(요강을 사용하는 일 포함)과 '
 '대소변 후에 뒤처리 시 다른 사람의 계속적인 도움이 필요한 상태, 또는 간헐적으로 자가 인공도뇨가 가능한 상태(CIC), 기저귀를 이용한 '
 '배뇨, 배변 상태(15%) · 화장실에 가는 일, 배변, 배뇨는 독립적으로 가능하나 대소변후 뒤처리에 있어 다 른 사람의 도움이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 149},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000989',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
