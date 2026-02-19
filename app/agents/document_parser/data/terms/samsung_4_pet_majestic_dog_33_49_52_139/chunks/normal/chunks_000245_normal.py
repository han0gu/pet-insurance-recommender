from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 계약자가 제1회 보험료를 신 용카드로 납입한 특별약관의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하며 이자를 '
 '더하여 지급하지 않습니다. ⑥ 회사가 제3항에 따라 일부보장 제외 조건을 붙여 승낙하였더라도 청약일로부터 5년'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000245',
              'chunk_char_len': 135,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
