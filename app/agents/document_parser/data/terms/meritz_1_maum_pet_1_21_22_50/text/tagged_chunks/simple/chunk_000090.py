from langchain_core.documents import Document

chunk = Document(
    page_content=('해지되었으나 계약자가 제33조(보험료의 환급)에 따라 보험료를 돌려받지 않은 경우 계\n'
 '약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을\n'
 '청약할 수 있습니다. 이 경우 회사가 그 청약을 승낙한 때에는 계약자는 부활(효력회\n'
 '복)을 청약한 날까지의 연체된 보험료에 보험개발원이 공시하는 월평균 정기예금이율- 16 -+ 1% 범위내에서 각 상품별로 회사가 정하는 '
 '이율로 계산한 금액을 더하여 납입하여\n'
 '야 합니다.② 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제15조(계약 전 알릴 의무), 제'),
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
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
