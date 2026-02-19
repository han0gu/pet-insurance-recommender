from langchain_core.documents import Document

chunk = Document(
    page_content=('제19조(계약내용의 변경 등)\n'
 '① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 서면 등으로 알 리거나 보험증권의 뒷면에 기재하여 '
 '드립니다.\n'
 '1. 보험종목 2. 보험기간 3. 보험료 납입주기, 납입방법 및 납입기간 4. 계약자, 피보험자 5. 보험가입금액, 보험료 등 기타 '
 '계약의 내용'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
