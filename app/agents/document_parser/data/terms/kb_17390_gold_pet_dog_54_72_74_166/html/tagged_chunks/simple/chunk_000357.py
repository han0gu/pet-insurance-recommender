from langchain_core.documents import Document

chunk = Document(
    page_content=(". 부재자의 생사가 5년간 분명하지 않은 때에는 법원은 이해관계인이나 검사</p><br><p id='12' "
 "data-category='paragraph' style='font-size:14px'>의 청구에 의하여 실종선고를 하여야 "
 "합니다.</p><br><p id='13' data-category='paragraph' style='font-size:14px'>2"),
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
 'indexing': {'chunk_id': 'chunk_000357',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
