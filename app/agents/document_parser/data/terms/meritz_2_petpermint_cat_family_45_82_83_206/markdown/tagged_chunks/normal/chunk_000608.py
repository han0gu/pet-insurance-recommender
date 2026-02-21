from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6) 정신행동에 경미한 장해를 남긴 때 | 10 |\n'
 '| 7) 극심한 치매 : CDR 척도 5점 | 100 |\n'
 '| 8) 심한 치매 : CDR 척도 4점 | 80 |\n'
 '| 9) 뚜렷한 치매 : CDR 척도 3점 | 60 |\n'
 '| 10) 약간의 치매 : CDR 척도 2점 | 40 |\n'
 '| 11) 심한 뇌전증 발작이 남았을 때 | 70 |\n'
 '| 12) 뚜렷한 뇌전증 발작이 남았을 때 | 40 |\n'
 '| 13) 약간의 뇌전증 발작이 남았을 때 | 10 |\n'
 '# 나. 장해판정기준# 1) 신경계- 가) “신경계에 장해를 남긴 때”라 함은 뇌, 척수'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000608',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
