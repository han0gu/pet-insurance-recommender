from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가보 험료(정산금액을 포함합니다)를 계약자가 '
 '납입하지 않았을 때, 회사는 위험이 증가되 기 전에 적용된 보험요율(이하 「변경전 요율」 이라 합니다)의 위험이 증가된 후에 적 용해야 '
 '할 보험요율(이하 「변경후 요율」 이라 합니다)에 대한 비율에 따라 보험금을 삭감하여 지급합니다. 다만, 증가된 위험과 관계없이 발생한 '
 '보험금 지급사유에 관해 서는 원래대로 지급합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000581',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
