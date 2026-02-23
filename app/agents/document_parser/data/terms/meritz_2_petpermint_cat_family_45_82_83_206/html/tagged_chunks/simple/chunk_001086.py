from langchain_core.documents import Document

chunk = Document(
    page_content=('척도 5점</td><td>100</td></tr><tr><td>8) 심한 치매 : CDR 척도 '
 '4점</td><td>80</td></tr><tr><td>9) 뚜렷한 치매 : CDR 척도 '
 '3점</td><td>60</td></tr><tr><td>10) 약간의 치매 : CDR 척도 '
 '2점</td><td>40</td></tr><tr><td>11) 심한 뇌전증 발작이 남았을 '
 '때</td><td>70</td></tr><tr><td>12) 뚜렷한 뇌전증 발작이 남았을 '
 '때</td><td>40</td></tr><tr><td>13) 약간의 뇌전증 발작이'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_001086',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
