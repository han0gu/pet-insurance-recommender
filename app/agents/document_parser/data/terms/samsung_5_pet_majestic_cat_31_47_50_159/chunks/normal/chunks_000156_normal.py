from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 최저보증이율은 연단위 복리 0.25%로 합니다. ② 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 해약환급금을 청구하여야 '
 '하 며, 회사는 청구를 접수한 날부터 3영업일 이내에 해약환급금을 지급합니다. 해약환급 금 지급일까지의 기간에 대한 이자의 계산은 '
 '보험금을 지급할 때의 적립이율 계산([별 표1] 참조)에 따릅니다. ③ 제10조(환급금의 중도인출) 제1항에 따라 환급금을 중도인출한 '
 '경우에는 중도인출금 및 중도인출금에 부리되었을 이자만큼 해약환급금에서 차감하여 계산하므로 제1항에 정한 지급금이 감소합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 46},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
