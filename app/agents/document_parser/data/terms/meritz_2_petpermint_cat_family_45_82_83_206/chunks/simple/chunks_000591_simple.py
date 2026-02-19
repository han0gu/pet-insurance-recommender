from langchain_core.documents import Document

chunk = Document(
    page_content=('NAA025 NAA026 | 골절 (뒷다리)\n'
 '2 | 눈 및 부속 기관의 질환 | AIA001 | 눈 및 부속 기관의 양성 신생물\n'
 'AIB001 | 눈 및 부속 기관의 악성 신생물\n'
 'AIC001 | 눈 및 부속 기관의 신생물 (양성 또는 악성이 불확실한)\n'
 'FAA001 | 안검 외반\n'
 'FAA002 | 안검 내반\n'
 'FAA003 | 안검염\n'
 'FAA004 | 다래끼 / 산립종 / 마이봄선종\n'
 'FAA005 | 체리아이 · 제3안검 돌출\n'
 'FAA006 | 비루관폐쇄\n'
 'FAA007 | 유루증'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 169},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000591',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
