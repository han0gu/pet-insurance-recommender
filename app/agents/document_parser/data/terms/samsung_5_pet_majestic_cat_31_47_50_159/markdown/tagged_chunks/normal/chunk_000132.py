from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 규정에 따라 해지하지 않은 계약은 파산선고 후 3개월이 지난 때에는 그 효\n'
 '- 력을 잃습니다.\n'
 '- ③ 제1항의 규정에 따라 계약이 해지되거나 제2항의 규정에 따라 계약이 효력을 잃는 경\n'
 '- 우에 회사는 제36조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.\n'
 '# 제36조 (해약환급금)- ① 이 약관에 따른 해약환급금은 “보험료 및 해약환급금 산출방법서”에 따라 계산합니\n'
 '- 다. 이 때 적립부분 순보험료에 대하여는 보험료 납입일(회사에 입금된 날을 말합니'),
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
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
