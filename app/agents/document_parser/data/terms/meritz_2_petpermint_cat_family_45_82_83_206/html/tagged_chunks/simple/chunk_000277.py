from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자간에 약정한 금액으로 보험사고 가 발생할 때 회사가 지급할 최대 보험금을 말 합니다.</td></tr><tr><td>자기 '
 '부담금</td><td>보험사고로 인하여 발생한 손해에 대하여 계약 자 또는 피보험자가 부담하는 일정 금액을 말 '
 '합니다.</td></tr><tr><td>보험금 분담</td><td>이 계약에서 보장하는 위험과 같은 위험을 보 장하는 다른 '
 '계약(공제계약을 포함합니다)이 있을 경우 비율에 따라 손해를 '
 '보상합니다.</td></tr><tr><td>공제계약</td><td>공제(미래에 발생할 수 있는 경제적 불안을 제'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000277',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
