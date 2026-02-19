from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고, 해지 시 회사가 환급하여야 할 보험료가 있을 경우에는 '
 '제33조(보험료의 환급)에 따른 보험 료를 계약자에게 지급합니다.\n'
 '제32조(회사의 파산선고와 해지)\n'
 '① 회사가 파산의 선고를 받은 때에는 계약자는 계약을 해지할 수 있습니다. ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 '
 '3개월이 지난 때에는 그 효력 을 잃습니다. ③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
