from langchain_core.documents import Document

chunk = Document(
    page_content=("id='68' data-category='list'></p><br><h1 id='69' "
 "style='font-size:14px'>제2조(보험금 지급에 관한 세부규정)</h1><br><h1 id='70' "
 "style='font-size:14px'>\uf000 제1조(보험금의 지급사유)의</h1><br><p id='71' "
 "data-category='paragraph' style='font-size:14px'>반려동물 위탁비용은 같은 상해의 치료를 목적으로 "
 "2</p><br><p id='72' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001252',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
