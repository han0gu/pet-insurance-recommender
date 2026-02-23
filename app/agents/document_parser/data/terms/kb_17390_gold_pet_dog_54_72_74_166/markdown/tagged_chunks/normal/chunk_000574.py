from langchain_core.documents import Document

chunk = Document(
    page_content=('| 백내장/녹내장수술 | 백내장/녹내장수술 | 연간1회한 | 50만원 |\n'
 '| 특정처치(이물제거) | 이물제거(내시경) | 연간2회한 | 200만원 |\n'
 '| 특정처치(이물제거) | 이물제거(구토유도약물) | 연간2회한 | 20만원 |\n'
 '| 창상/교상치료 | 창상/교상치료 | 연간1회한 | 70만원 |\n'
 '| 특정약물치료Ⅱ | 특정약물치료Ⅱ | 연간12회한 | 10만원 |\n'
 '| 특정재활치료Ⅱ | 특정재활치료Ⅱ | 연간12회한 | 5만원 |\n'
 '| 항암약물치료 | 항암약물치료 | 연간6회한 | 30만원 |'),
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
 'indexing': {'chunk_id': 'chunk_000574',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
