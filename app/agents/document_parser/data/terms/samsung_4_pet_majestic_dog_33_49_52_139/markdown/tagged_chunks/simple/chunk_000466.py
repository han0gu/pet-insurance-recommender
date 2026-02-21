from langchain_core.documents import Document

chunk = Document(
    page_content=('을 말합니다.- 1. 보험증권에 기재된 피보험자(이하 「피보험자 본인」 이라 합니다)\n'
 '- 2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 배우자(이하 「배우자」 라\n'
 '- 합니다)\n'
 '3. 피보험자 본인 또는 배우자와 생계를 같이 하고, 보험증권에 기재된 주택의 주민등록\n'
 '상 동거중인 동거 친족(민법 제 777조)\n'
 '4. 피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀-'),
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
 'indexing': {'chunk_id': 'chunk_000466',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
