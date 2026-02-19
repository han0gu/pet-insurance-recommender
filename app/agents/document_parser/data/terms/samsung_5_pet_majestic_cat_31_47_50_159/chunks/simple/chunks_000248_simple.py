from langchain_core.documents import Document

chunk = Document(
    page_content=('제20조 (청약의 철회)\n'
 '① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만, 회사가 건강상태 진단을 지원하는 계약, '
 '보험기간이 90일 이내인 계약 또는 전문금융 소비자가 체결한 계약은 청약을 철회할 수 없습니다.\n'
 '<용어풀이>\n'
 '[전문금융소비자]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 56},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000248',
              'chunk_char_len': 156,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
