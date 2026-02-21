from langchain_core.documents import Document

chunk = Document(
    page_content=('우 회사는 이 특별약관의 해약환급금을 계약자에게 지급합니다. 다만, 타인을 위한 계약\n'
 '의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 특별\n'
 '약관을 해지할 수 있습니다.# 제 25조 (중대사유로 인한 해지)① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 1개월 '
 '이내에 이 특별\n'
 '약관을 해지할 수 있습니다.- 1. 계약자 또는 피보험자가 보험금을 지급받을 목적으로 고의로 보험금 지급사유를\n'
 '- 발생시킨 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000377',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
