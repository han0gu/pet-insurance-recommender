from langchain_core.documents import Document

chunk = Document(
    page_content=('- 인이 보험수익자로 지정된 경우에는 제1항의 통지를 계약자에게 할 수 있습니다.\n'
 '- ④ 회사는 제1항의 통지를 계약이 해지된 날부터 7일 이내에 하여야 합니다.\n'
 '- ⑤ 보험수익자는 통지를 받은 날(제3항에 따라 계약자에게 통지된 경우에는 계약자가 통\n'
 '- 지를 받은 날을 말합니다)부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.\n'
 '# 제6관 계약의 해지 및 해약환급금 등# 제 32조 (계약자의 임의해지 및 피보험자의 서면동의 철회권)- ① 계약자는 계약이 소멸하기 '
 '전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사는'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
