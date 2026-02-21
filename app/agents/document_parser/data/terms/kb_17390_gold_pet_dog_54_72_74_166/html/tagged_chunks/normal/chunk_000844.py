from langchain_core.documents import Document

chunk = Document(
    page_content=("경우에</p><br><p id='16' data-category='paragraph' style='font-size:18px'>- 104 "
 "-</p><p id='17' data-category='paragraph' style='font-size:16px'>회사는 보험금을 "
 '지급하지 않으며, 계약 전 알릴 의무 위반사실(계약해지 등의<br>원인이 되는 위반사실을 구체적으로 명시)뿐만 아니라 계약 전 알릴 '
 '의무사항이<br>중요한 사항에 해당되는 사유를 "반대증거가 있는 경우 이의를 제기할 수 있습니<br>다"라는 문구와 함께 계약자에게'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000844',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
