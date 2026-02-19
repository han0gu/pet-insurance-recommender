from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자가 제1회 보험료를 신 용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하며 이자 를 더하여 '
 '지급하지 않습니다. ⑤ 회사가 제2항에 따라 일부보장 제외 조건을 붙여 승낙하였더라도 청약일로부터 5년 (갱신계약의 경우에는 최초계약 '
 '청약일로부터 5년)이 지나는 동안 보장이 제외되는 질 병으로 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없을 경우, 청약일로부터 '
 '5 년이 지난 이후에는 이 약관에 따라 보장합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000087',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
