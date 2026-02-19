from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험안내자료의 내용이 약관의 내 용과 다른 경우에는 계약자에게 유리한 내용으로 '
 '계약이 성립된 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000141',
              'chunk_char_len': 91,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
