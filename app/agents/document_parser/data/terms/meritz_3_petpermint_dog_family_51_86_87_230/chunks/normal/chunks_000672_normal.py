from langchain_core.documents import Document

chunk = Document(
    page_content=('NAA007 | 슬개골 (아) 탈구- (좌측-1기)\n'
 'NAA008 | 슬개골 (아) 탈구- (좌측-2,3,4기)\n'
 'NAA009 | 슬개골 (아) 탈구- (우측-1기)\n'
 'NAA010 | 슬개골 (아) 탈구- (우측-2,3,4기)\n'
 'NAA011 NAA012 | 십자 인대 손상 파열 (전방 / 후방) (좌측) 십자 인대 손상 파열 (전방 / 후방) (우측)\n'
 'NAA013 | 골절 (뒷다리) (좌측)\n'
 'NAA014 | 골절 (뒷다리) (우측)\n'
 'NAA015 | 성장판 골절 (뒷다리)\n'
 'NAA016 | 관절염 (뒷다리)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 195},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000672',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
