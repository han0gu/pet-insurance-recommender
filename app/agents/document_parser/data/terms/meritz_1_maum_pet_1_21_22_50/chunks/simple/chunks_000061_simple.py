from langchain_core.documents import Document

chunk = Document(
    page_content=('제17조(알릴 의무 위반의 효과)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 보험금 지급사유의 발생여부에 관계없이 그 사실을 안 날부터 1개월 이내에 이 계약을 해지할 '
 '수 있습니다.\n'
 '1. 계약자, 피보험자 또는 이들의 대리인이 고의 또는 중대한 과실로 제15조(계약 전 알릴 의무)를 위반하고 그 의무가 중요한 사항에 '
 '해당하는 경우 2. 뚜렷한 위험의 증가와 관련된 제16조(계약 후 알릴 의무) 제1항에서 정한 계약 후 알릴 의무를 계약자, 피보험자 '
 '또는 이들의 대리인의 고의 또는 중대한 과실로 이행 하지 않았을때'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 10},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000061',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
