from langchain_core.documents import Document

chunk = Document(
    page_content=("단, 상해(상해로 인한 창상 또는 교상, 이물섭취를 포함)를 직접적인 원인으</h1><br><h1 id='219' "
 "style='font-size:16px'>로 치료를 받은 경우에는 보장개시일은 보험계약일로 합니다.</h1><h1 id='220' "
 "style='font-size:16px'>예 시 2</h1><br><h1 id='221' "
 "style='font-size:16px'>백내장/녹내장수술, 특정약물치료Ⅱ, 항암약물치료</h1><br><figure "
 "id='222'><img style='font-size:16px'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001010',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
