from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기타 보험회사가 필요하다고 인정하는 서류 및 보험수익자가 보험금의 수령에 반<br>필요하여 제출하는 서류 '
 "려동<br>물</p><br><p id='74' data-category='paragraph' "
 "style='font-size:14px'>병</p><p id='75' data-category='paragraph' "
 "style='font-size:16px'>제5조(특별약관의 소멸)</p><br><p id='76' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는"),
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
 'indexing': {'chunk_id': 'chunk_001093',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
