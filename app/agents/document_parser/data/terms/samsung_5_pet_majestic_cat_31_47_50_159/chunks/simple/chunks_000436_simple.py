from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 지급금액\n'
 '안면부 5cm이상 성형수술시 | 이 특별약관 보험가입금액의 50%\n'
 '안면부 10cm이상 성형수술시 | 이 특별약관 보험가입금액의 50%\n'
 '<예시안내>\n'
 '[수술길이에 따른 보험금 지급]\n'
 '(기준 : 이 특별약관의 보험가입금액 100만원)\n'
 '안면부의 성형수술길이 | 보험금 지급사유 | 지급보험금\n'
 '13cm | 5cm이상 성형수술, 10cm이상 성형수술 모두에 해당 | 안면부 5cm이상 성형수술비 + 안면부 10cm이상 성형수술비 = '
 '50만원 + 50만원 = 100만원'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000436',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
