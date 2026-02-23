from langchain_core.documents import Document

chunk = Document(
    page_content=(". 법<br>ㆍ</p><br><p id='112' data-category='paragraph' "
 "style='font-size:20px'>규정</p><p id='113' data-category='paragraph' "
 "style='font-size:16px'>별표16 특정정신질환 분류표</p><p id='114' "
 "data-category='paragraph' style='font-size:18px'>- 163 -</p><br><p id='115' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001799',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
