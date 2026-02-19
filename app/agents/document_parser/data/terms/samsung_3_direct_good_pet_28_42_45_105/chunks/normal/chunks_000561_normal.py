from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 회사는 그러 한 보증을 제공할 책임은 부담하지 않습니다. 마. 피보험자가 제9조(손해배상청구에 대한 회사의 해결) 제2항 및 '
 '제3항의 회사의 요구에 따르기 위하여 지출한 비용'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 87},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 103,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
