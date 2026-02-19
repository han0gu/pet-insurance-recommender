from langchain_core.documents import Document

chunk = Document(
    page_content=('제20조(청약의 철회)\n'
 '\uf000 일반금융소비자인 계약자는 보험증권을 받은 날부터 15 일 이내에 그 청약을 철회할 수 있습니다. 다만, 회사가 건 강상태 '
 '진단을 지원하는 계약, 보험기간이 90일 이내인 계 약 또는 전문금융소비자가 체결한 계약은 청약을 철회할 수 없습니다.\n'
 '【일반금융소비자】\n'
 '전문금융소비자가 아닌 계약자를 말합니다.\n'
 '【전문금융소비자】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 68},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
