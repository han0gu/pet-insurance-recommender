from langchain_core.documents import Document

chunk = Document(
    page_content=('복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자가 제1회 보험료를 신 용카드로 납입한 특별약관의 승낙을 거절하는 경우에는 '
 '신용카드의 매출을 취소하며 이자를 더하여 지급하지 않습니다.\n'
 '제 15조 (사기에 의한 계약)\n'
 '계약자 또는 피보험자의 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에는 계약체결일부터 5년 이내(사기사실을 안 날부터 1개월 '
 '이내)에 계약을 취소할 수 있습니 다.\n'
 '제 16조 (특별약관의 무효)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000598',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
