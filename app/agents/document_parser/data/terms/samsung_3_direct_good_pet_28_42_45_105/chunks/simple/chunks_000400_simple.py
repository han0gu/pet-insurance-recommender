from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑧ 제22조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))에 따라 이 특별 약관이 부활(효력회복)된 경우에는 '
 '부활(효력회복)계약을 제2항의 최초계약으로 봅니 다. 부활(효력회복)이 여러차례 발생된 경우에는 각각의 부활(효력회복)계약을 최초계 '
 '약으로 봅니다.'),
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
 'indexing': {'chunk_id': 'chunk_000400',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
