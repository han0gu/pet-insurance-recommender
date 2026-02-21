from langchain_core.documents import Document

chunk = Document(
    page_content=('| 입 원 의 료 비 Ⅲ | 입원 중 수술을 하지 않은 날의 경우 | MRI,CT 및 내시경처치 를 받은 날의 경우 | 연간 두번째 이상 '
 '| 1일당 3만원/ 5만원 중 보험증 권에 기재된 자기부 담금 | 1일당 30만원 |\n'
 '| 입 원 의 료 비 Ⅲ | 입원 중 수술을 하지 않은 날의 경우 | MRI,CT 및 내시경처치를 받지 않은 날의 경우 | MRI,CT '
 '및 내시경처치를 받지 않은 날의 경우 | 1일당 3만원/ 5만원 중 보험증 권에 기재된 자기부 담금 | 1일당 30만원 |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000413',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
