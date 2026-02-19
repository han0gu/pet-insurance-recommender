from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 지급금액\n'
 '안면부 5cm이상 성형수술시 | 이 특별약관 보험가입금액의 50%\n'
 '안면부 10cm이상 성형수술시 | 이 특별약관 보험가입금액의 50%\n'
 '<예시안내>\n'
 '[수술길이에 따른 보험금 지급]\n'
 '안면부의 성형수술길이 | 보험금 지급사유 | 지급보험금\n'
 '13cm | 5cm이상 성형수술, 10cm이상 성형수술 모두에 해당 | 안면부 5cm이상 성형수술비 + 안면부 10cm이상 성형수술비 = '
 '50만원 + 50만원 = 100만원\n'
 '7cm | 5cm이상 성형수술에 해당 | 5cm이상 성형수술비 = 50만원'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 83},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000439',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
