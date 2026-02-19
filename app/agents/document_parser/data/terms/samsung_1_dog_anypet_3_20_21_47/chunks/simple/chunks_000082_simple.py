from langchain_core.documents import Document

chunk = Document(
    page_content=('. 【국세 및 지방세 체납처분 절차】 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 체 납된 세금에 대하여 가산금 '
 '징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다. 국세 및 지 방세 체납시 국세청 및 지방자치단체에 의해 채무자의 '
 '환급금이 압류될 수 있으며, 체납처분 절차에 따라 회사는 채권자에게 환급금을 지급하게 됩니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000082',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
