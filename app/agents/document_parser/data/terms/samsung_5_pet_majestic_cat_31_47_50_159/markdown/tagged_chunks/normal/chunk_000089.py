from langchain_core.documents import Document

chunk = Document(
    page_content=('계산한 금액을 더하여 지급합니다.# 제23조 (계약의 무효)다음 중 한 가지에 해당하는 경우에는 계약을 무효로 하며 이미 납입한 보험료를 '
 '돌려 드\n'
 '립니다. 다만, 회사의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무효\n'
 '임을 알았거나 알 수 있었음에도 보험료를 반환하지 않은 경우에는 보험료를 납입한 날\n'
 '의 다음날부터 반환일까지의 기간에 대하여 회사는 이 계약의 보험계약대출이율을 연단\n'
 '위 복리로 계산한 금액을 더하여 돌려 드립니다.- 1. 타인의 사망을 보험금 지급사유로 하는 계약에서 계약을 체결할 때까지 피보험자'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000089',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
