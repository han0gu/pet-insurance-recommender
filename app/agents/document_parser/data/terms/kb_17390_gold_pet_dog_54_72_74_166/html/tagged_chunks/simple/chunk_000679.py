from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제1항 내지 제2항에 해당하지 않는 시술(체외 충격파 쇄석술 및 변연절제를</p><br><p id='238' "
 "data-category='list'></p><br><p id='239' data-category='paragraph' "
 "style='font-size:16px'>동반하지 않은 단순 창상봉합술 등)<br>\uf000 제1항에서 의료기관이라 함은 의료법 "
 "제3조(의료기관) 제2항에서 정한 국내의 병</p><br><table id='240'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000679',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
