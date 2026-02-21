from langchain_core.documents import Document

chunk = Document(
    page_content=("계약자 또는 보험수익자가</p><br><p id='103' data-category='paragraph' "
 "style='font-size:14px'>2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니</p><p id='104' "
 "data-category='paragraph' style='font-size:16px'>- 58 -</p><p id='105' "
 "data-category='paragraph' style='font-size:16px'>다"),
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
 'indexing': {'chunk_id': 'chunk_000083',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
