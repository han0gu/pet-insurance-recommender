from langchain_core.documents import Document

chunk = Document(
    page_content=('된 피보험자 및 그 가족을 말합니다.\n'
 '① 보험증권에 기재된 피보험자(이하「피보험자 본인」이 라 합니다) ② 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 '
 '배우자(이하「배우자」라 합니다) ③ 피보험자 본인 또는 배우자와 생계를 같이 하는 동거 친족 및 별거 중인 미혼자녀\n'
 '【민법 제777조(친족의 범위)】\n'
 '친족관계로 인한 법률상 효력은 이 법 또는 다른 법률에 특별한 규정이 없는 한 다음 각호에 해당하는 자에 미친 다.\n'
 '1. 8촌이내의 혈족 2. 4촌이내의 인척 3. 배우자'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 186},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000631',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
