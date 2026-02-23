from langchain_core.documents import Document

chunk = Document(
    page_content=('원인으로 보험계약의 보험금 지급사유가 발생하였을 경우에는 보험계약의 규정에\n'
 '도 불구하고 계약을 체결할 때 정한 삭감기간에 따라 다음과 같이 보험금을 지급\n'
 '합니다.| 경과기간 | 기준 | 삭감기간별 보험금지급비율 | 삭감기간별 보험금지급비율 | 삭감기간별 보험금지급비율 | 삭감기간별 '
 '보험금지급비율 | 삭감기간별 보험금지급비율 |\n'
 '| --- | --- | --- | --- | --- | --- | --- |\n'
 '| 경과기간 | 기준 | 1년 | 2년 | 3년 | 4년 | 5년 |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000714',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
