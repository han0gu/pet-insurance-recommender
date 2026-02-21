from langchain_core.documents import Document

chunk = Document(
    page_content=('부터 2년이 지난 후에 성형수술이 가능하다는 진단을 받은 경우에는 그 진단으로 대\n'
 '체할 수 있습니다)을 받은 경우 아래에 정한 금액을 안면부 상해흉터복원(성형) 수술\n'
 '비로 보험수익자에게 지급합니다.| 구 분 | 지급금액 |\n'
 '| --- | --- |\n'
 '| 안면부 5cm이상 성형수술시 | 이 특별약관 보험가입금액의 50% |\n'
 '| 안면부 10cm이상 성형수술시 | 이 특별약관 보험가입금액의 50% |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000367',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
