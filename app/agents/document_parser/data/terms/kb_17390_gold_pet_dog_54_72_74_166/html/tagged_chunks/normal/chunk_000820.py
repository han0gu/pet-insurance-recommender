from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반</p><br><p id='188' data-category='paragraph' style='font-size:16px'>관 련 "
 "법 규 상법</p><br><p id='189' data-category='paragraph' style='font-size:16px'>∙ "
 "상법 제651조(고지의무위반으로 인한 계약해지)</p><br><p id='190' data-category='paragraph' "
 "style='font-size:14px'>보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하 여<br>중요한 사항을"),
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
 'indexing': {'chunk_id': 'chunk_000820',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
