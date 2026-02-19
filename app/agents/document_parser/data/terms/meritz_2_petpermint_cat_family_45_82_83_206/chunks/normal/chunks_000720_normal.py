from langchain_core.documents import Document

chunk = Document(
    page_content=('. 5) “발가락뼈 일부를 잃었을 때”라 함은 첫째 발가락 에서는 지관절, 다른 네 발가락에서는 제1지관절(근 위지관절)부터 심장에서 먼 '
 '쪽으로 발가락 뼈 일부가 절단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나 발가락 길이의 단축 없이 골편만 떨어진 상태는 해당하 지 '
 '않는다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 198},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000720',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
