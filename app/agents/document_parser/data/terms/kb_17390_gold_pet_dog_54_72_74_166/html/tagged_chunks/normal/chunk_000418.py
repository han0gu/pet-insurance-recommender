from langchain_core.documents import Document

chunk = Document(
    page_content=("id='89' data-category='paragraph' style='font-size:20px'>상해흉터복원수술비</p><br><p "
 "id='90' data-category='paragraph' style='font-size:20px'>5.</p><p id='91' "
 "data-category='paragraph' style='font-size:16px'>제1조(보험금의 지급사유)</p><br><p "
 "id='92' data-category='paragraph' style='font-size:14px'>상</p><br><p"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000418',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
