from langchain_core.documents import Document

chunk = Document(
    page_content=('손상</td><td>S00-S09</td></tr><tr><td>목의 '
 '손상</td><td>S10-S19</td></tr><tr><td>여러 신체부위를 포함하는 손상 중,</td><td>\u3000'
 '</td></tr><tr><td>\u3000목과 함께 머리를 침범한 표재성 '
 '손상</td><td>T00.0</td></tr><tr><td>\u3000기타 신체부위를 복합적으로 침범한 표재성 '
 '손상</td><td>T00.8주)</td></tr><tr><td>\u3000목과 함께 머리를 침범한 열린 상처 '
 '신체부위를</td><td>T01.0'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001693',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
