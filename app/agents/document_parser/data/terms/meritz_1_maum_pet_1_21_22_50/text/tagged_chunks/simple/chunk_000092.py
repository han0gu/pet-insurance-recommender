from langchain_core.documents import Document

chunk = Document(
    page_content=('청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약 포함) 제15조(계약 전 알\n'
 '릴 의무)를 위반한 경우에는 제17조(알릴 의무 위반의 효과)가 적용됩니다.【부활(효력회복)】보험료 납입을 연체하여 계약이 해지되고 '
 '계약자가 해약환급금을 받지 않은 경우\n'
 '회사가 정하는 소정의 절차에 따라 해지된 계약을 다시 되살리는 것을 말합니다.제29조(강제집행 등의 절차에 따라 해지된 계약의 '
 '특별부활(효력회복))① 타인을 위한 계약의 경우 제33조(보험료의 환급)에 따른 계약자의 환급금 청구권에 대'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000092',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
