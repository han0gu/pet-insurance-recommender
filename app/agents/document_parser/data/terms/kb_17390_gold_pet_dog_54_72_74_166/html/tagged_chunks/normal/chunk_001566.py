from langchain_core.documents import Document

chunk = Document(
    page_content=("지속되는 경우</p><p id='63' data-category='paragraph' style='font-size:14px'>146 "
 "KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><p id='64' data-category='list'></p><br><p "
 "id='65' data-category='list'></p><p id='66' data-category='paragraph' "
 "style='font-size:14px'>7.</p><br><h1 id='67' style='font-size:14px'>체간골의"),
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
 'indexing': {'chunk_id': 'chunk_001566',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
