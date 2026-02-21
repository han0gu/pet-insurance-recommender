from langchain_core.documents import Document

chunk = Document(
    page_content=('- 익자를 변경할 수 있습니다.\n'
 '- ⑦ 회사는 제1항 제4호에 따라 계약자를 변경한 경우, 변경된 계약자에게 보험증권 및 약\n'
 '- 관을 드리고, 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니\n'
 '- 다.\n'
 '# 제25조 (보험나이 등)- ① 이 약관에서의 피보험자의 나이는 보험나이를 기준으로 합니다. 다만, 제23조(계약의\n'
 '- 무효) 제2호의 경우에는 실제 만 나이를 적용합니다.\n'
 '- ② 제1항의 보험나이는 계약일 현재 피보험자의 실제 만 나이를 기준으로 6개월 미만의'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
