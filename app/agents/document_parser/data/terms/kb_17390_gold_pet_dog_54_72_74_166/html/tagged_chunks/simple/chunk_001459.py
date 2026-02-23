from langchain_core.documents import Document

chunk = Document(
    page_content=("규정을</p><br><p id='132' data-category='paragraph' "
 "style='font-size:16px'>따릅니다.</p><br><p id='133' data-category='paragraph' "
 "style='font-size:14px'>별</p><br><p id='134' data-category='paragraph' "
 "style='font-size:14px'>특</p><p id='135' data-category='paragraph' "
 "style='font-size:14px'>약</p><br><p"),
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
 'indexing': {'chunk_id': 'chunk_001459',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
