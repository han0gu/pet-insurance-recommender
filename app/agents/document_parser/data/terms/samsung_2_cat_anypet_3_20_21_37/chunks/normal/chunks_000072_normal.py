from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제23조[보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지]에 따라 계약이 해지되었으 나 계약자가 제30조(보험료의 '
 '환급)에 따라 보험료를 돌려받지 않는 경우 계약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 '
 '수 있습니다. 이 경우 회사 가 그 청약을 승낙한 때에는 계약자는 부활(효력회복)을 청약한 날까지의 연체된 보험료에 보험개 발원이 '
 '공시하는 월평균 정기예금이율 +1% 범위내에서 각 상품별로 회사가 정하는 이율로 계산한 금액을 더하여 납입하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 15},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000072',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
