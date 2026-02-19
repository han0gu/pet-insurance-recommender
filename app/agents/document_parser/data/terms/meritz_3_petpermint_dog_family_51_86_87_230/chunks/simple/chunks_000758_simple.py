from langchain_core.documents import Document

chunk = Document(
    page_content='신경 장해”란 추간판 탈출증이 확인되고 신경생리검사에서 명확한 신경근병 증의 소견이 지속되는 경우',
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 213},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000758',
              'chunk_char_len': 54,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.92}},
)
