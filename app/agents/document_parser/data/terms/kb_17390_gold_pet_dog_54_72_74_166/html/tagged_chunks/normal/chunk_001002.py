from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">치 료 구 분</td><td>대 상 원 인</td></tr><tr><td '
 'colspan="2">MRI/CT</td><td>상해 또는 질병</td></tr><tr><td '
 'colspan="2">백내장/녹내장수술</td><td>백내장 또는 녹내장</td></tr><tr><td rowspan="2">특정처치 '
 '(이물제거)</td><td>이물제거(내시경)</td><td '
 'rowspan="2">이물섭취</td></tr><tr><td>이물제거(구토유도약물)</td></tr><tr><td'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001002',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
