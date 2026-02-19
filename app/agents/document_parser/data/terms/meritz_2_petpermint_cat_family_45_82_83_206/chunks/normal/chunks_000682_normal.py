from langchain_core.documents import Document

chunk = Document(
    page_content='신경 장해”란 추간판 탈출증이 확인되고 신경생리검사에서 명확한 신경근병 증의 소견이 지속되는 경우',
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 188},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000682',
              'chunk_char_len': 54,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
