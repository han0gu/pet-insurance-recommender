from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>\uf000</h1><br><p id='55' data-category='list' "
 "style='font-size:14px'>보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.<br>1"),
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
 'indexing': {'chunk_id': 'chunk_000042',
              'chunk_char_len': 128,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
