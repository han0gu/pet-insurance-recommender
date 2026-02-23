from langchain_core.documents import Document

chunk = Document(
    page_content=('(공시이율의 적용 및 공시) 제1항에 따라 변경된 이율을 적용하며, 최저보증이율은 연\n'
 '단위 복리 0.25%로 합니다.<용어풀이># [최저보증이율]운용자산이익률 및 시중금리가 하락하더라도 회사에서 보증하는 최저한도의 '
 '적용이율입니다. 예를\n'
 '들어, 적립금이 공시이율에 따라 부리되며 공시이율이 0.1%인 경우(최저보증이율이 공시이율보다\n'
 '큰 경우), 적립금은 공시이율(0.1%)이 아닌 최저보증이율로 부리됩니다.- ② 회사는 계약자 및 보험수익자의 청구에 의하여 제1항에 '
 '의한 만기환급금을 지급하는'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000039',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
