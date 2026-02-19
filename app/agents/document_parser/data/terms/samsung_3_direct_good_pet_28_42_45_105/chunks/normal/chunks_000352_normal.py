from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험증권에 기재된 피보험자(이하「피보험자 본인」이라 합니다) 2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 '
 '배우자(이하「배우자」라 합니다)\n'
 '3. 피보험자 본인 또는 배우자와 생계를 같이 하고, 보험증권에 기재된 주택의 주민등록\n'
 '상 동거중인 동거 친족(민법 제 777조)\n'
 '4. 피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀\n'
 '<관련법규>\n'
 '[민법 제777조(친족의 범위)에서 규정한 친족의 범위]\n'
 ': 8촌 이내의 혈족, 4촌 이내의 인척, 배우자\n'
 '제 6조 (보험금을 지급하지 않는 사유)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000352',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
