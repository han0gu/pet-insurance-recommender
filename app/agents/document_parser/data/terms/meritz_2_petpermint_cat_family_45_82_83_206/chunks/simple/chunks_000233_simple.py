from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 이 특별약관의 청약을 받고, 제1회 보험료를 받 은 경우에 건강진단을 받지 않는 계약은 청약일, 진단계약 은 '
 '진단일(재진단의 경우에는 최종 진단일)부터 30일 이내 에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다. 그러나 30일 '
 '이내에 승낙 또는 거절의 통지가 없 으면 승낙된 것으로 봅니다. \uf000 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 95},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000233',
              'chunk_char_len': 212,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
