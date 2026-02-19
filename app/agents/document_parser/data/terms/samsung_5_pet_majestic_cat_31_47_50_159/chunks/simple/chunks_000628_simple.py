from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자가 통 지를 받은 날을 말합니다)부터 15일 이내에 '
 '제1항의 절차를 이행할 수 있습니다.\n'
 '제24조 (계약자의 임의해지)\n'
 '계약자는 특별약관이 소멸하기 전에는 언제든지 이 특별약관을 해지할 수 있으며, 이 경 우 회사는 이 특별약관의 해약환급금을 계약자에게 '
 '지급합니다. 다만, 타인을 위한 계약 의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 특별 약관을 해지할 '
 '수 있습니다.\n'
 '제25조 (중대사유로 인한 해지)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000628',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
