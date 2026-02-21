from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관을 드리고, 변경된 계약자가 요청하는 경우 약관의 중요한 내용을 설명하여 드립니\n'
 '- 다.\n'
 '# 제22조 (보험나이 등)- ① 이 약관에서의 피보험자의 나이는 보험나이를 기준으로 합니다. 다만, 제20조(계약의\n'
 '- 무효) 제2호의 경우에는 실제 만 나이를 적용합니다.\n'
 '- ② 제1항의 보험나이는 계약일 현재 피보험자의 실제 만 나이를 기준으로 6개월 미만의\n'
 '- 끝수는 버리고 6개월 이상의 끝수는 1년으로 하여 계산하며, 이후 매년 계약해당일에\n'
 '- 나이가 증가하는 것으로 합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000079',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
