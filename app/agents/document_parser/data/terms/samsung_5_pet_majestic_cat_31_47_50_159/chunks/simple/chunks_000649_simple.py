from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '「반려묘 수술비(치과및 구강질환포함)(재가입형) 확대보장」 에 대한 보장개시일(책임개시일) 계 산]\n'
 '주) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 합니 다.\n'
 '<유의사항>\n'
 "[수술] 동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상 해 또는 질병 치료를 위하여 "
 '수의사법 제 17조(개설)에서 규정한 국내의 동물병원에서 수의사의'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 107},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000649',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
