from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[「반려견 사망위로금」에 대한 보장개시일(책임개시일) 계산]\n'
 '제2조 (보험금을 지급하지 않는 사유)\n'
 '회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 85},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000539',
              'chunk_char_len': 100,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
