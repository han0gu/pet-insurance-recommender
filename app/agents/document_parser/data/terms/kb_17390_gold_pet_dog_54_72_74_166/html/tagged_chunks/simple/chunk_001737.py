from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>법</p><br><p id='67' data-category='paragraph' "
 "style='font-size:14px'>ㆍ</p><br><p id='68' data-category='paragraph' "
 "style='font-size:20px'>규정</p><table id='69' "
 "style='font-size:14px'><thead><tr><td "
 'colspan="2"></td><td></td></tr></thead><tbody><tr><td rowspan="27">창상봉합술Ⅱ '
 '(급여)'),
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
 'indexing': {'chunk_id': 'chunk_001737',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
