from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자 의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가 수신되지 않은 것을 확인한 '
 '경우에는 서면(등기우편 등)으로 다시 알려드립니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 72},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000398',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
