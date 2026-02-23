from langchain_core.documents import Document

chunk = Document(
    page_content=('니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상\n'
 '책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.<용어풀이>[공제계약]\n'
 '유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되는 계약을 말합니다.\n'
 '우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.이 계약의 보상책임액\n'
 '손해액 ×\n'
 '다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액- 124② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 '
 '회사의 제1항에 의한'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000672',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
