from langchain_core.documents import Document

chunk = Document(
    page_content=('【설명】 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기 재하였거나 그 서류 또는 증거를 위조 '
 '또는 변조한 경우 회사는 그 사실을 안 날부터 1개월 이내에 계 약을 해지할 수 있습니다. 다만, 이 경우에도 회사는 이미 발생한 보험금 '
 '지급사유에 대해서는 보험금 을 지급합니다.\n'
 '② 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 통지하고 제30조(보험료의 환급)에 따라 보험료를 계약자에게 '
 '지급합니다.\n'
 '제28조(회사의 파산선고와 해지)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
