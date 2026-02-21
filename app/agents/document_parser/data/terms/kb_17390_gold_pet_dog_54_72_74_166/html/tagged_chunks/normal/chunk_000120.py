from langchain_core.documents import Document

chunk = Document(
    page_content=('. 뚜렷한 위험의 증가와 관련된 제15조(상해보험계약 후 알릴 의무) 제1항에서<br>정한 계약 후 알릴 의무를 계약자 또는 피보험자의 '
 '고의 또는 중대한 과실로<br>이행하지 않았을때<br>제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 '
 "계약</p><br><p id='150' data-category='paragraph' "
 "style='font-size:14px'>\uf000</p><br><p id='151' data-category='paragraph' "
 "style='font-size:14px'>을 해지할 수"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000120',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
