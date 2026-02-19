from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[계약해당일 계산]\n'
 '최초계약일과 동일한 월, 일을 말합니다. 계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일 단, 계약해당일 2월 '
 '29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.\n'
 '제19조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000413',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
