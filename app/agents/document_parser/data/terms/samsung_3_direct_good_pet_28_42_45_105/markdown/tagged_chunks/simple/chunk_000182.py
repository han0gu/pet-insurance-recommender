from langchain_core.documents import Document

chunk = Document(
    page_content=('과함에 따라 위험의 크기 및 정도가 점차 증가하는 위험 또는 기간의 경과에 상관없이 일정한 상태\n'
 '를 유지하는 위험에 적용하는 방법으로 위험 정도에 따라 특별보험료를 추가로 부가하는 방법을\n'
 '말합니다.- ④ 회사는 이 특별약관의 청약을 받고 제1회 보험료를 받은 경우에 건강진단을 받지 않\n'
 '- 는 특별약관은 청약일, 진단계약은 진단일(재진단의 경우에는 최종진단일)부터 30일\n'
 '- 이내에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다. 그러나 30일\n'
 '- 이내에 승낙 또는 거절의 통지가 없으면 승낙된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000182',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
