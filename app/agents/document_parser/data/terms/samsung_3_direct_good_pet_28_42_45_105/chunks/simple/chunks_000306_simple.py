from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험안내자료]\n'
 '계약의 청약을 권유하기 위해 만든 자료 등을 말합니다. [기명날인] 자기 이름을 쓰고 도장을 찍는 것을 말합니다.\n'
 '제 43조 (회사의 손해배상책임)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000306',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
