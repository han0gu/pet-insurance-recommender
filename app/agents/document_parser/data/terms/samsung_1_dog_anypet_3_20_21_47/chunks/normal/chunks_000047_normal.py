from langchain_core.documents import Document

chunk = Document(
    page_content=('제16조(청약의 철회)\n'
 '① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만, 의무보험의 경우에는 철회의사를 표시한 시점에 '
 '동종의 다른 의무보험에 가입된 경우에만 철회할 수 있으며, 보험기간이 90일 이내인 계약 또는 전문금융소비자가 체결한 계약은 청약을 '
 '철회할 수 없습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 173,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
