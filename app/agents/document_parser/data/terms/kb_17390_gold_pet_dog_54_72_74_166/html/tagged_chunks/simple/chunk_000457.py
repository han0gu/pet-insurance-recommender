from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 제1조(보험금의 지급사유)의</p><br><p id='154' "
 "data-category='paragraph' style='font-size:16px'>외모특정상해(머리,목)수술비는 같은 상해로 두 "
 "종</p><br><p id='155' data-category='paragraph' "
 "style='font-size:14px'>상</p><p id='156' data-category='list' "
 "style='font-size:14px'>반<br>류 이상의 외모특정상해(머리,목)수술을 받거나 같은"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000457',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
