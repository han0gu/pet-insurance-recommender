from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA009 | 슬개골 (아) 탈구- (우측-1기) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA010 | 슬개골 (아) 탈구- (우측-2,3,4기) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA011 NAA012 | 십자 인대 손상 파열 (전방 / 후방) (좌측) 십자 인대 파열 (전방 '
 '/ (우측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA013 | 손상 후방) 골절 (뒷다리) (좌측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA014 | 골절 (뒷다리) (우측) |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000471',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
