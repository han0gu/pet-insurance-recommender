from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제7조(계약 전 알릴 의무)의 규정에 따라 계약자 또는 피보험자가 회사에 알</p><br><p id='49' "
 "data-category='list' style='font-size:14px'>린 내용이 보험금 지급사유의 발생에 영향을 미쳤음을 회사가 "
 '증명하는 경우<br>2'),
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
 'indexing': {'chunk_id': 'chunk_000875',
              'chunk_char_len': 152,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
