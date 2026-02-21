from langchain_core.documents import Document

chunk = Document(
    page_content=('나 계약자가 제30조(보험료의 환급)에 따라 보험료를 돌려받지 않는 경우 계약자는 해지된 날부터\n'
 '3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약할 수 있습니다. 이 경우 회사\n'
 '가 그 청약을 승낙한 때에는 계약자는 부활(효력회복)을 청약한 날까지의 연체된 보험료에 보험개\n'
 '발원이 공시하는 월평균 정기예금이율 +1% 범위내에서 각 상품별로 회사가 정하는 이율로 계산한'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
