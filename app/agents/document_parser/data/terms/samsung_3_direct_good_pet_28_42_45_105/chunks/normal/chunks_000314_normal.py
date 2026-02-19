from langchain_core.documents import Document

chunk = Document(
    page_content=('- 59 -\n'
 '59 / 181\n'
 '※ 약관에서 인용된 법·규정은「별표 및 참고」의 「약관에서 인용된 법·규정」에서 확인할 수 있습니다.\n'
 '1. 상해 관련 특별약관\n'
 '제1관 일반사항'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 62},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000314',
              'chunk_char_len': 95,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
