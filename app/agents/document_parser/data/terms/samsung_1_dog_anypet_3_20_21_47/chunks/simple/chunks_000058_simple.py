from langchain_core.documents import Document

chunk = Document(
    page_content=('제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게 돌려드리며, 보험 료를 받은 기간에 대하여 보험개발원이 '
 '공시하는 보험계약대출이율을 연단위 복리로 계산한 금액 을 더하여 지급합니다.\n'
 '제18조(계약의 무효)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000058',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
