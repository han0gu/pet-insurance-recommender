from langchain_core.documents import Document

chunk = Document(
    page_content=("id='97' data-category='paragraph' style='font-size:16px'>- 98 -</p><p "
 "id='98' data-category='paragraph' style='font-size:20px'>특별약관</p><p id='99' "
 "data-category='paragraph' style='font-size:16px'>제4장 반려동물 관련 특별약관</p><p "
 "id='100' data-category='paragraph' style='font-size:14px'>- 99 -</p><h1 "
 "id='101'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000752',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
