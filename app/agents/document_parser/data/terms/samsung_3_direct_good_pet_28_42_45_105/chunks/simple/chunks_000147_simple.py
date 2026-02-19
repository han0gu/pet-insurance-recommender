from langchain_core.documents import Document

chunk = Document(
    page_content=('특별약관\n'
 '※ 약관에서 인용된 법·규정은 「별표 및 참고」 의 「약관에서 인용된 법·규정」 에서 확인할 수 있습니다.\n'
 '특별약관 일반사항\n'
 '제1관 목적 및 용어의 정의\n'
 '제1조(목적)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 45},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
