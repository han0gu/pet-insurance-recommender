from langchain_core.documents import Document

chunk = Document(
    page_content=('제 26조 (계약의 소멸)\n'
 '① 회사가 제3조(보험금의 지급사유)에서 정한 상해 후유장해(80%이상) 보험금을 지급한 때에는 그 손해보상의 원인이 생긴 때부터 이 '
 '계약은 소멸되며 그 때부터 효력이 없 습니다. ② 피보험자가 보험기간 중에 사망하였을 경우에는 “보험료 및 해약환급금 산출방법 서”에 '
 '정하는 바에 따라 회사가 적립한 사망당시의 계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 계약은 더 이상 효력이 없습니다.\n'
 '<용어풀이>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
