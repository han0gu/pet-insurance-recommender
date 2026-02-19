from langchain_core.documents import Document

chunk = Document(
    page_content=('제17조(약관 교부 및 설명의무 등)\n'
 '① 회사는 계약자가 청약할 때에 계약자에게 약관의 중요한 내용을 설명하여야 하며, 청약 후에 다음 각 호의 방법 중 계약자가 원하는 '
 '방법을 확인하여 지체 없이 약관 및 계약자 보관용 청약서를 제'),
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
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
