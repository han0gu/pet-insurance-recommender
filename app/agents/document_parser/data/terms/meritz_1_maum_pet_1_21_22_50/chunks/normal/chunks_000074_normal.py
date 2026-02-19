from langchain_core.documents import Document

chunk = Document(
    page_content=('제20조(청약의 철회)\n'
 '① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만, 회사가 건강상태 진단을 지원하는 계약, '
 '보험기간이 90일 이내인 계약 또는 전문금융소 비자가 체결한 계약은 청약을 철회할 수 없습니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
