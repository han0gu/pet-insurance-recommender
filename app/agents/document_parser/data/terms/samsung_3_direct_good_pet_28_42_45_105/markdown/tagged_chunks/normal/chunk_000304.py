from langchain_core.documents import Document

chunk = Document(
    page_content=('을 말합니다.- 1. 보험증권에 기재된 피보험자(이하「피보험자 본인」이라 합니다)\n'
 '- 2. 피보험자 본인의 가족관계등록상 또는 주민등록상에 기재된 배우자(이하「배우자」라\n'
 '- 합니다)'),
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
 'indexing': {'chunk_id': 'chunk_000304',
              'chunk_char_len': 101,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
