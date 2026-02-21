from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우에 회사는 제35조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.\n'
 '# 제 35조 (해약환급금)① 이 약관에 따른 해약환급금은 “보험료 및 해약환급금 산출방법서”에 따라 계산하며,\n'
 '계약이 해지될 경우에는 아래와 같이 해약환급금을 지급합니다.1. 해약환급금 구분이 해약환급금 일부지급형일 때에는 보험료 납입기간 중 '
 '계약이\n'
 '해지될 경우 표준형 상품 해약환급금의 50%에 해당하는 금액을 지급하며, 보험료\n'
 '납입이 완료되고 보험료 납입기간이 종료된 이후 계약이 해지될 경우 표준형 상품'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
