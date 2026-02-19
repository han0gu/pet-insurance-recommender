from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 갱신계약의 보험기간은 갱신전 계약의 보험기간과 동일하게 적용하며, 갱신계 약의 갱신은 회사가 사업방법서에서 정한 갱신형 계약의 '
 '갱신종료나이 계약해 당일까지로 합니다. 나. 가.목에도 불구하고 갱신일부터 회사가 사업방법서에서 정한 갱신종료나이의 계약해당일까지가 '
 '가.목의 보험기간 미만일 경우 그 잔여기간을 보험기간으로 하여 갱신되는 것으로 하며, 세부사항은 회사의 사업방법서를 따릅니다. 다. '
 '동일한 사고에 대하여 갱신전 계약에서 이미 보험금 지급사유가 발생하여 해당 보험금이 지급된 경우에는 갱신계약에서 보상하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000812',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
