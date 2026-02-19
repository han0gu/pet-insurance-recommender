from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험증권에 기재된 피보험자(이하 「피보험자 본인」 이라 합니다) 2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 '
 '배우자(이하 「배우자」 라 합니다) 3. 피보험자 본인 또는 배우자와 생계를 같이 하고, 보험증권에 기재된 주택의 주민등록 상 동거중인 '
 '동거 친족(민법 제 777조) 4. 피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀\n'
 '<관련법규>\n'
 '[민법 제777조(친족의 범위)에서 규정한 친족의 범위]\n'
 ': 8촌 이내의 혈족, 4촌 이내의 인척, 배우자\n'
 '제6조 (보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000542',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
