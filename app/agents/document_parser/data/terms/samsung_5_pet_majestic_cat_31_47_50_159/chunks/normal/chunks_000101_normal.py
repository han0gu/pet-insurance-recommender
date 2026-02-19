from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게 돌려\n'
 '드리며, 보험료를 받은 기간에 대하여 이 계약의 보험계약대출이율을 연단위 복리로 계산한 금액을 더하여 지급합니다.\n'
 '제23조 (계약의 무효)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 40},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000101',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
