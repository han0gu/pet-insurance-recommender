from langchain_core.documents import Document

chunk = Document(
    page_content=('함께 머리를 침범한 열린 상처 신체부위를</td><td>T01.0 T01.8주)</td></tr><tr><td>\u3000기타 복합적으로 '
 '침범한 열린 상처 \u3000목과 함께 머리를 침범한 골절</td><td>T02.0</td></tr><tr><td>\u3000기타 '
 '신체부위를 복합적으로 침범한 골절</td><td>T02.8주)</td></tr><tr><td>\u3000목과 함께 머리를 침범한 '
 '탈구,염좌 및 긴장</td><td>T03.0</td></tr><tr><td>\u3000기타 신체부위를 복합적으로 침범한 탈구,염좌 및'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_001694',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
