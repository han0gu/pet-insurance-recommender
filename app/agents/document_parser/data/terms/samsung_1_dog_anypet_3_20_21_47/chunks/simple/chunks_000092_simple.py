from langchain_core.documents import Document

chunk = Document(
    page_content=('제28조(회사의 파산선고와 해지)\n'
 '① 회사가 파산의 선고를 받은 때에는 계약자는 계약을 해지할 수 있습니다. ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 '
 '3개월이 지난 때에는 그 효력을 잃습니다. ③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경우에 '
 '회사는 제30조(보험료의 환급)에 의한 보험료를 계약자에게 지급합니다.\n'
 '제29조(위법계약의 해지)'),
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
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
