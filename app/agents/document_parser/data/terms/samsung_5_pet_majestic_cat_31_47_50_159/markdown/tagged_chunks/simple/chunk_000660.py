from langchain_core.documents import Document

chunk = Document(
    page_content=('금융위원회의 명령에 따라 약관이 개정된 경우에는 갱신일 현재의 약관을 적용합\n'
 '니다.# 2. 갱신시 보험기간의 운영가. 갱신계약의 보험기간은 갱신전 계약의 보험기간과 동일하게 적용하며, 갱신계\n'
 '약의 갱신은 회사가 사업방법서에서 정한 갱신형 계약의 갱신종료나이 계약해\n'
 '당일까지로 합니다.\n'
 '나. 가.목에도 불구하고 갱신일부터 회사가 사업방법서에서 정한 갱신종료나이의\n'
 '계약해당일까지가 가.목의 보험기간 미만일 경우 그 잔여기간을 보험기간으로\n'
 '하여 갱신되는 것으로 하며, 세부사항은 회사의 사업방법서를 따릅니다.'),
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
 'indexing': {'chunk_id': 'chunk_000660',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
