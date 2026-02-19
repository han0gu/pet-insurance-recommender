from langchain_core.documents import Document

chunk = Document(
    page_content=('제29조(위법계약의 해지)\n'
 '① 계약자는 「금융소비자보호에 관한 법률」 제47조 및 관련규정이 정하는 바에 따라 계약체결에 대한 회사의 법위반사항이 있는 경우 '
 '계약체결일부터 5년 이내의 범위에서 계약자가 위반사항을 안 날 부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하여 위법계약의 해지를 '
 '요구할 수 있습니다. 다만, 의무보험의 해지를 요구하려는 경우에는 동종의 다른 의무보험에 가입되어 있어야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
