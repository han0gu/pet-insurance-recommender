from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 30일 이 내에 승낙 또는 거절의 통지가 없으면 승낙된 것으로 봅니다. ④ 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 '
 '거절통지와 함께 받은 금액을 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율+1%를 연단위 복리로 계산한 금액을 더하여 '
 '지급합니다. 다만, 회사는 계약자가 제1회 보험료를 신 용카드로 납입한 특별약관의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하며 '
 '이자를 더하여 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000595',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
