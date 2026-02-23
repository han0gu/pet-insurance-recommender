from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 남긴 때 | 10∼100 |\n'
 '| 2) 정신행동에 극심한 장해를 남긴때 | 100 |\n'
 '| 3) 정신행동에 심한 장해를 남긴 때 | 75 |\n'
 '| 4) 정신행동에 뚜렷한 장해를 남긴 때 | 50 |\n'
 '| 5) 정신행동에 약간의 장해를 남긴 때 | 25 |\n'
 '| 6) 정신행동에 경미한 장해를 남긴 때 | 10 |\n'
 '| 7) 극심한 치매 : CDR 척도 5점 | 100 |\n'
 '| 8) 심한 치매 : CDR 척도 4점 | 80 |\n'
 '| 9) 뚜렷한 치매 : CDR 척도 3점 | 60 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000681',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
