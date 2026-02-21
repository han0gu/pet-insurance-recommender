from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2 | 눈 및 부속 기관의 질환 | FAA001 | 안검 외반 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA002 | 안검 내반 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA003 | 안검염 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA004 | 다래끼 / 산립종 / 마이봄선종 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA005 | 체리아이 · 제3안검 돌출 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA006 | 비루관폐쇄 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA007 | 유루증 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000559',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
