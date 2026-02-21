from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 159</p><br><p id='64' "
 "data-category='paragraph' style='font-size:14px'>별</p><br><p id='65' "
 "data-category='paragraph' style='font-size:14px'>표</p><p id='66' "
 "data-category='paragraph' style='font-size:14px'>법</p><br><p id='67'"),
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
 'indexing': {'chunk_id': 'chunk_001736',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
