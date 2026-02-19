from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 신경계․정신행동 장해의 경우 ① 개호(장해로 혼자서 활동이 어려운 사람을 곁에서 돌 보는 것) 여부 ② 객관적 이유 및 개호의 '
 '내용을 추가 로 기재하여야 한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 177},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000624',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
