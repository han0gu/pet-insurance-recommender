from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 증가된 위험과 관계없이 발생한 보험금 지급사유에 관해서는 원래대로 지급합니다. ⑤ 계약자 또는 피보험자가 고의 또는 중대한 '
 '과실로 제1항 각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요율이 변경전 요율보다 높을 때에는 회사는 그 변경사실을 안 날부터 '
 '1개월 이내에 계약자 또는 피보험자에게 제4항에 의해 보장됨을 통보하고 이에 따라 보험금을 지급합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000060',
              'chunk_char_len': 208,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
